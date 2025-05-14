import csv
import numpy as np
import cvxpy as cp
import qutip  # if needed
import logging

# -------------------- SeeSaw Algorithm Functions -------------------- #

def generate_random_wishart(d):
    """
    Generate a random d×d positive semidefinite matrix via the Wishart construction.
    A random complex matrix A is generated and then W = A A† is returned.
    """
    A = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    return A @ A.conj().T

def random_povm(n_outcomes, d):
    """
    Generate a valid POVM for a d-dimensional Hilbert space with n_outcomes.
    Each POVM element is built from a random positive operator (Wishart matrix)
    and then normalized via the inverse square root of the sum so that the elements
    sum to the identity.
    """
    Ws = [generate_random_wishart(d) for _ in range(n_outcomes)]
    S = sum(Ws)
    eigvals, eigvecs = np.linalg.eigh(S)
    inv_sqrt_S = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
    return [inv_sqrt_S @ W @ inv_sqrt_S for W in Ws]

def random_POVM_partie(n_settings, n_outcomes, d):
    """
    For a given party (Alice or Bob), generate a list of POVMs (one per measurement setting)
    using the random_povm function.
    """
    return [random_povm(n_outcomes, d) for _ in range(n_settings)]

def construc_G(inequality_line, POVM_alice, POVM_bob, scenario_tuple, d):
    """
    Construct the operator G from the provided inequality coefficients (CG notation) and
    the POVMs of Alice and Bob. The coefficients are split into joint and marginal terms.
    """
    X, Y, A, B = scenario_tuple
    numbers = list(map(float, inequality_line.strip().split()))
    n_joint_prob = (A - 1) * X * (B - 1) * Y
    n_alice_prob = (A - 1) * X
    n_bob_prob = (B - 1) * Y
    if len(numbers) != n_joint_prob + n_alice_prob + n_bob_prob:
        raise Exception(f"Expected {n_joint_prob+n_alice_prob+n_bob_prob} coefficients, got {len(numbers)}.")
    joint_prob = numbers[0:n_joint_prob]
    alice_marginal = numbers[n_joint_prob:n_joint_prob + n_alice_prob]
    bob_marginal = numbers[n_joint_prob + n_alice_prob:]
    
    G = 0
    count = 0
    # Alice marginals
    for a in range(A - 1):
        for x in range(X):
            G = G + alice_marginal[count] * np.kron(POVM_alice[x][a], np.eye(d))
            count += 1
    count = 0
    # Bob marginals
    for b in range(B - 1):
        for y in range(Y):
            G = G + bob_marginal[count] * np.kron(np.eye(d), POVM_bob[y][b])
            count += 1
    count = 0
    # Joint terms
    for a in range(A - 1):
        for b in range(B - 1):
            for x in range(X):
                for y in range(Y):
                    G = G + joint_prob[count] * np.kron(POVM_alice[x][a], POVM_bob[y][b])
                    count += 1
    return G

def optimize_rho_TB(G, d):
    """
    Optimize the bipartite state ρ to maximize Tr(Gρ) subject to:
      - ρ is positive semidefinite.
      - ρ has unit trace.
      - ρ satisfies the PPT condition (partial transpose on Bob's subsystem).
    """
    d_total = d**2
    rho = cp.Variable((d_total, d_total), hermitian=True)
    sigma = cp.Variable((d_total, d_total), hermitian=True)
    
    constraints = [
        rho >> 0,
        sigma >> 0,
        cp.trace(rho) == 1
    ]
    for iA in range(d):
        for iB in range(d):
            for jA in range(d):
                for jB in range(d):
                    orig_row = iA * d + iB
                    orig_col = jA * d + jB
                    pt_row = iA * d + jB   # Swap Bob's indices.
                    pt_col = jA * d + iB
                    constraints.append(sigma[pt_row, pt_col] == rho[orig_row, orig_col])
    
    objective = cp.Maximize(cp.real(cp.trace(G @ rho)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, warm_start=False)
    return rho.value

def general_kron(a, b):
    """
    Compute the Kronecker product of matrices a and b and return as a CVXPY expression.
    """
    expr = np.kron(a, b)
    num_rows = expr.shape[0]
    rows = [cp.hstack(expr[i, :]) for i in range(num_rows)]
    return cp.vstack(rows)

def SDP_alice(bob_POVM, rho, inequality_line, scenario_tuple, d):
    """
    Optimize Alice's POVM (decision variables) via SDP with Bob's POVM fixed.
    """
    X, Y, A, B = scenario_tuple
    alice_POVM = []
    for _ in range(X):
        setting = []
        for _ in range(A):
            setting.append(cp.Variable((d, d), hermitian=True))
        alice_POVM.append(setting)
    
    numbers = list(map(float, inequality_line.strip().split()))
    n_joint_prob = (A - 1) * X * (B - 1) * Y
    n_alice_prob = (A - 1) * X
    if len(numbers) != n_joint_prob + n_alice_prob + (B - 1) * Y:
        raise Exception("Inequality coefficients do not match the expected number.")
    joint_prob = numbers[0:n_joint_prob]
    alice_marginal = numbers[n_joint_prob : n_joint_prob + n_alice_prob]
    
    G = 0
    count = 0
    for a in range(A - 1):
        for x in range(X):
            G = G + alice_marginal[count] * general_kron(alice_POVM[x][a], np.eye(d))
            count += 1
    count = 0
    for a in range(A - 1):
        for b in range(B - 1):
            for x in range(X):
                for y in range(Y):
                    G = G + joint_prob[count] * general_kron(alice_POVM[x][a], bob_POVM[y][b])
                    count += 1

    obj = cp.Maximize(cp.real(cp.trace(G @ rho)))
    constraints = []
    for setting in alice_POVM:
        for meas in setting:
            constraints.append(meas >> 0)
        constraints.append(sum(setting) == np.eye(d))
    
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, warm_start=False)
    
    optimized_alice = []
    for x in range(X):
        setting = []
        for a in range(A):
            setting.append(alice_POVM[x][a].value)
        optimized_alice.append(setting)
    return optimized_alice

def SDP_bob(alice_POVM, rho, inequality_line, scenario_tuple, d):
    """
    Optimize Bob's POVM (decision variables) via SDP with Alice's POVM fixed.
    """
    X, Y, A, B = scenario_tuple
    bob_POVM = []
    for _ in range(Y):
        setting = []
        for _ in range(B):
            setting.append(cp.Variable((d, d), hermitian=True))
        bob_POVM.append(setting)
    
    numbers = list(map(float, inequality_line.strip().split()))
    n_joint_prob = (A - 1) * X * (B - 1) * Y
    n_alice_prob = (A - 1) * X
    if len(numbers) != n_joint_prob + n_alice_prob + (B - 1) * Y:
        raise Exception("Inequality coefficients do not match the expected number.")
    joint_prob = numbers[0:n_joint_prob]
    bob_marginal = numbers[n_joint_prob + n_alice_prob:]
    
    G = 0
    count = 0
    for b in range(B - 1):
        for y in range(Y):
            G = G + bob_marginal[count] * general_kron(np.eye(d), bob_POVM[y][b])
            count += 1
    count = 0
    for a in range(A - 1):
        for b in range(B - 1):
            for x in range(X):
                for y in range(Y):
                    G = G + joint_prob[count] * general_kron(alice_POVM[x][a], bob_POVM[y][b])
                    count += 1
    
    obj = cp.Maximize(cp.real(cp.trace(G @ rho)))
    constraints = []
    for setting in bob_POVM:
        for meas in setting:
            constraints.append(meas >> 0)
        constraints.append(sum(setting) == np.eye(d))
    
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, warm_start=True)
    
    optimized_bob = []
    for y in range(Y):
        setting = []
        for b in range(B):
            setting.append(bob_POVM[y][b].value)
        optimized_bob.append(setting)
    return optimized_bob

def SeeSaw(scenario, inequality_line, d, n=1000):
    """
    Implements the full SeeSaw algorithm by iteratively optimizing over Alice's POVM,
    Bob's POVM, and the state ρ. For each random initialization, the algorithm alternates
    between SDP optimization of Alice's and Bob's POVMs and updating ρ via an SDP with PPT
    constraints until convergence.
    
    Returns:
        tuple: (global_max, global_alice, global_bob) corresponding to the best achieved value
               and the corresponding POVMs.
    """
    X, Y, A, B = scenario
    global_max = -10
    global_alice, global_bob = None, None
    
    for init in range(n):
        # Random initialization.
        alice_povm = random_POVM_partie(X, A, d)
        bob_povm = random_POVM_partie(Y, B, d)
        G = construc_G(inequality_line, alice_povm, bob_povm, scenario, d)
        rho = optimize_rho_TB(G, d)
        maximal_result = np.trace(rho @ G)
        maximal_alice = alice_povm
        maximal_bob = bob_povm
        iter_values = [maximal_result.real]
        count = 0
        
        while True:
            alice_povm = SDP_alice(bob_povm, rho, inequality_line, scenario, d)
            bob_povm = SDP_bob(alice_povm, rho, inequality_line, scenario, d)
            G = construc_G(inequality_line, alice_povm, bob_povm, scenario, d)
            rho = optimize_rho_TB(G, d)
            result = np.trace(rho @ G)
            iter_values.append(result.real)
            
            if result >= maximal_result:
                maximal_result = result
                maximal_alice = alice_povm
                maximal_bob = bob_povm
            if count >= 15 or abs(iter_values[-1] - iter_values[-2]) / max(abs(iter_values[-1]), 1) <= 1e-9:
                break
            count += 1
        print(f"Initialization {init}: Iteration values = {iter_values}")
        if maximal_result > global_max:
            global_max = maximal_result
            global_alice = maximal_alice
            global_bob = maximal_bob
        if init % 10 == 0:
            print(init, global_max.real)
    return global_max, global_alice, global_bob

def partial_transpose(rho, dims, subsystem='B'):
    """
    Return the partial transpose of density matrix ρ on the given subsystem.
    """
    dA, dB = dims
    rho_4d = rho.reshape((dA, dB, dA, dB))
    if subsystem == 'B':
        rho_pt = rho_4d.transpose(0, 3, 2, 1).reshape(rho.shape)
    elif subsystem == 'A':
        rho_pt = rho_4d.transpose(2, 1, 0, 3).reshape(rho.shape)
    else:
        raise ValueError("subsystem must be 'A' or 'B'")
    return rho_pt

def check_density_matrix(rho, tol=1e-9, verbose=False):
    valid = True
    if not np.allclose(rho, rho.conj().T, atol=tol):
        max_diff = np.max(np.abs(rho - rho.conj().T))
        print(f"Warning: ρ is not Hermitian (max diff = {max_diff:.3e}).")
        valid = False
    elif verbose:
        print("ρ is Hermitian.")
    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) < -tol:
        print(f"Warning: ρ is not PSD (min eigenvalue = {np.min(eigvals):.3e}).")
        valid = False
    elif verbose:
        print(f"ρ is PSD (min eigenvalue = {np.min(eigvals):.3e}).")
    trace_val = np.trace(rho).real
    if abs(trace_val - 1.0) > tol:
        print(f"Warning: Tr(ρ) = {trace_val:.6f} (should be 1).")
        valid = False
    elif verbose:
        print(f"Tr(ρ) = {trace_val:.6f}.")
    return valid

def check_ppt_condition(rho, dims, tol=1e-9, verbose=False):
    rho_pt = partial_transpose(rho, dims, subsystem='B')
    eigenvals = np.linalg.eigvalsh(rho_pt)
    min_eig = np.min(eigenvals)
    if verbose:
        print(f"PPT eigenvalues: {eigenvals}")
    if min_eig < -tol:
        print(f"Warning: PPT condition violated (min eigenvalue = {min_eig:.3e}).")
    else:
        if verbose:
            print("PPT condition satisfied.")

def check_povms(povm_sets, tol=1e-9, verbose=False):
    all_valid = True
    for idx, elements in enumerate(povm_sets):
        dim = elements[0].shape[0]
        sum_op = np.zeros((dim, dim), dtype=complex)
        for j, E in enumerate(elements):
            if not np.allclose(E, E.conj().T, atol=tol):
                print(f"Warning: POVM set {idx}, element {j} not Hermitian.")
                all_valid = False
            vals = np.linalg.eigvalsh(E)
            if np.min(vals) < -tol:
                print(f"Warning: POVM set {idx}, element {j} not PSD (min eigen = {np.min(vals):.3e}).")
                all_valid = False
            sum_op = sum_op + E
        I = np.eye(dim)
        if not np.allclose(sum_op, I, atol=tol):
            diff = np.linalg.norm(sum_op - I)
            print(f"Warning: POVM set {idx} does not sum to identity (||ΣE - I|| = {diff:.3e}).")
            all_valid = False
        if verbose:
            print(f"POVM set {idx}: Sum operator trace = {np.trace(sum_op):.3f}")
    return all_valid

def test_seesaw_algorithm(scenario, inequality_line, d, n=1000, max_iter=15, tol=1e-6, verbose=False):
    print("Running SeeSaw algorithm...")
    global_max, alice_POVM, bob_POVM = SeeSaw(scenario, inequality_line, d, n)
    G = construc_G(inequality_line, alice_POVM, bob_POVM, scenario, d)
    rho = optimize_rho_TB(G, d)
    print("\n--- Validity Checks ---")
    print("Checking density matrix ρ...")
    dm_valid = check_density_matrix(rho, verbose=verbose, tol=tol)
    print("Checking PPT condition on ρ...")
    check_ppt_condition(rho, (d, d), verbose=verbose, tol=tol)
    print("Checking POVMs for Alice...")
    alice_valid = check_povms(alice_POVM, verbose=verbose, tol=tol)
    print("Checking POVMs for Bob...")
    bob_valid = check_povms(bob_POVM, verbose=verbose, tol=tol)
    if dm_valid and alice_valid and bob_valid:
        print("All validity checks passed.")
    else:
        print("Some validity checks FAILED. Please review the warnings above.")
    return rho, alice_POVM, bob_POVM

def parse_scenario(scenario_str):
    return tuple(int(c) for c in scenario_str.strip())

# -------------------- Main Execution -------------------- #

if __name__ == "__main__":
    # Load the grouped results from CSV
    grouped_file = "consolidated_results.csv"
    records = []
    with open(grouped_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records.append(row)
    
    # Process only records with scenario "6322" (i.e. tuple (6,3,2,2))
    output_file = "seesaw_results_6322_dim3.txt"
    with open(output_file, "w") as f:
        d = 3  # Hilbert space dimension set to 3
        n_initializations = 1000  # 1000 random seeds/initializations
        
        # Process records in the order they appear in the CSV
        for rec in records:
            if rec["scenario"].strip() == "6322":
                scenario_str = rec["scenario"].strip()
                inequality_line = rec["inequality"]
                scenario_tuple = parse_scenario(scenario_str)
                
                f.write("========================================\n")
                f.write(f"Scenario: {scenario_str}\n")
                f.write(f"Inequality: {inequality_line}\n")
                print(f"\nRunning SeeSaw for scenario {scenario_str} with inequality:")
                print(inequality_line)
                
                # Run the SeeSaw algorithm
                rho, alice_POVM, bob_POVM = test_seesaw_algorithm(scenario_tuple, inequality_line, d,
                                                                   n=n_initializations, max_iter=15,
                                                                   tol=1e-6, verbose=False)
                # Recompute G and violation
                G = construc_G(inequality_line, alice_POVM, bob_POVM, scenario_tuple, d)
                violation = np.trace(rho @ G).real
                f.write(f"Final Violation: {violation}\n")
                
                f.write("Alice POVMs:\n")
                for x, setting in enumerate(alice_POVM):
                    for a, E in enumerate(setting):
                        f.write(f"  Setting {x}, Outcome {a}:\n")
                        f.write(np.array2string(E, precision=3))
                        f.write("\n")
                
                f.write("Bob POVMs:\n")
                for y, setting in enumerate(bob_POVM):
                    for b, F in enumerate(setting):
                        f.write(f"  Setting {y}, Outcome {b}:\n")
                        f.write(np.array2string(F, precision=3))
                        f.write("\n")
                f.write("\n")
                print("SeeSaw run complete for this inequality. Results saved.\n")
    
    print(f"All SeeSaw results for scenario '6322' with d=3 saved to {output_file}.")
