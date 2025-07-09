import numpy as np
import qutip
import cvxpy as cp
import sys
from toqito.rand import random_povm
import concurrent.futures as fut

MOSEK_PARAMS = {
    "MSK_IPAR_NUM_THREADS": 1,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10
}

def random_POVM_partie(n_settings, n_outcomes, d=4):
    #Sort random a POVM for one partie and return as a list such that: POVM[x][a] = M_{a|x}
    POVM = []
    povm_ndarray = random_povm(d, n_settings, n_outcomes)
    POVM = [[povm_ndarray[:, :, x, a] for a in range(n_outcomes)] for x in range(n_settings)]
    return POVM

def general_kron(a, b):
    """
    Returns a CVXPY Expression representing the Kronecker product of a and b.
    
    At most one of "a" and "b" may be CVXPY Variable objects.
    
    :param a: 2D numpy ndarray, or a CVXPY Variable with a.ndim == 2
    :param b: 2D numpy ndarray, or a CVXPY Variable with b.ndim == 2
    """
    if isinstance(a, cp.Expression) or isinstance(b, cp.Expression):
        if not isinstance(a, cp.Expression):
            a = cp.Constant(a)
        if not isinstance(b, cp.Expression):
            b = cp.Constant(b)
        return cp.kron(a, b)
    return np.kron(a, b)

def construct_G(alice_povm, bob_povm, d, general_kron_var=False, vertesi=True):
    # if vertesi Contruct the bell operator G from the family of inequalities described in https://arxiv.org/pdf/1704.08600
    # Uses the general kron function if we are dealing with an CVXPY experession
    if vertesi:
        if not general_kron_var:
            G = (d - 2) * (np.kron(alice_povm[0][0], bob_povm[1][0]) - np.kron(alice_povm[0][0], bob_povm[0][0]))
            for i in range(1, d):
                G += -(np.kron(np.identity(d), bob_povm[1][0]) - np.kron(alice_povm[i][0], bob_povm[1][0]))
                for j in range(1, d):
                    if i != j:
                        G += -np.kron(alice_povm[i][0], bob_povm[0][j])
        else:
            G = (d - 2) * (general_kron(alice_povm[0][0], bob_povm[1][0]) - general_kron(alice_povm[0][0], bob_povm[0][0]))
            for i in range(1, d):
                G += -(general_kron(np.identity(d), bob_povm[1][0]) - general_kron(alice_povm[i][0], bob_povm[1][0]))
                for j in range(1, d):
                    if i != j:
                        G += -general_kron(alice_povm[i][0], bob_povm[0][j])
    else:
        G = 0
    return G

def ptrace_A(mat, d):
    result = np.zeros((d, d), dtype=complex)
    reshaped = mat.reshape(d, d, d, d)
    for i in range(d):
        result += reshaped[i, :, i, :]
    return result

def ptrace_B(mat, d):
    result = np.zeros((d, d), dtype=complex)
    reshaped = mat.reshape(d, d, d, d)
    for j in range(d):
        result += reshaped[:, j, :, j]
    return result

def SDP_alice(bob_POVM, rho, d=4):
    # Define variables to be optimized
    X = d
    A = 2
    alice_POVM = []
    for _ in range(X):
        alice_POVM_x = []
        for _ in range(A):
            alice_POVM_x.append(cp.Variable((d, d), hermitian=True))
        alice_POVM.append(alice_POVM_x)
    # Define objective function: 
    B10 = bob_POVM[1][0]
    B0 = bob_POVM[0]
    coeff = []
    I_d = np.identity(d)
    coeff.append((d - 2) * ptrace_B(np.kron(I_d, B10 - B0[0]) @ rho, d))
    for i in range(1, d):
        s = B10 - sum(B0[j] for j in range(1, d) if j != i)
        coeff.append(ptrace_B(np.kron(I_d, s) @ rho, d))
    expr = 0
    for x in range(X):
        expr += cp.real(cp.trace(alice_POVM[x][0] @ coeff[x]))
    obj = cp.Maximize(expr)
    PSD_constraints = []
    for settings in alice_POVM:
        for measurement in settings:
            PSD_constraints.append(measurement >> 0)
    POVM_constraints = []
    for settings in alice_POVM:
        S = 0
        for measurement in settings:
            S += measurement
        POVM_constraints.append(S - np.identity(d) == 0)
    constraints = PSD_constraints + POVM_constraints
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS)
    optimized_alice_POVM = []
    for x in range(X):
        povm_x = []
        for a in range(A):
            povm_x.append(alice_POVM[x][a].value)
        optimized_alice_POVM.append(povm_x)
    return optimized_alice_POVM

def SDP_bob(alice_POVM, rho, d=4):
    # Define variables to be optimized
    bob_POVM = []
    bob_POVM_d_outcome = []
    for _ in range(d):
        bob_POVM_d_outcome.append(cp.Variable((d, d), hermitian=True))
    bob_POVM.append(bob_POVM_d_outcome)
    bob_POVM_two_outcome = []
    for _ in range(2):
        bob_POVM_two_outcome.append(cp.Variable((d, d), hermitian=True))
    bob_POVM.append(bob_POVM_two_outcome)
    # Define objective function:
    A0 = alice_POVM[0][0]
    As = [alice_POVM[i][0] for i in range(1, d)]
    I_d = np.identity(d)
    S10 = (d - 2) * A0 + sum(As) - (d - 1) * I_d
    coeff_10 = ptrace_A(np.kron(S10, I_d) @ rho, d)
    coeff_0 = []
    coeff_0.append(ptrace_A(np.kron(-(d - 2) * A0, I_d) @ rho, d))
    for j in range(1, d):
        coeff_0.append(ptrace_A(np.kron(-sum(As[k] for k in range(d - 1) if k != j - 1), I_d) @ rho, d))
    expr = cp.real(cp.trace(bob_POVM[1][0] @ coeff_10))
    for j in range(d):
        expr += cp.real(cp.trace(bob_POVM[0][j] @ coeff_0[j]))
    obj = cp.Maximize(expr)
    PSD_constraints = []
    for settings in bob_POVM:
        for measurement in settings:
            PSD_constraints.append(measurement >> 0)
    POVM_constraints = []
    for settings in bob_POVM:
        S = 0
        for measurement in settings:
            S += measurement
        POVM_constraints.append(S - np.identity(d) == 0)
    constraints = POVM_constraints + PSD_constraints
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS)
    optimized_bob_POVM = []
    povm_d_outcome = []
    for b in range(d):
        povm_d_outcome.append(bob_POVM[0][b].value)
    optimized_bob_POVM.append(povm_d_outcome)
    povm_two_outcome = []
    for b in range(2):
        povm_two_outcome.append(bob_POVM[1][b].value)
    optimized_bob_POVM.append(povm_two_outcome)
    return optimized_bob_POVM

def optimize_rho_TB(G, d=4):
    d_total = d ** 2
    rho = cp.Variable((d_total, d_total), hermitian=True)
    rho_pt = cp.partial_transpose(rho, dims=[d, d], axis=1)
    constraints = [rho >> 0, rho_pt >> 0, cp.trace(rho) == 1]
    objective = cp.Maximize(cp.real(cp.trace(G @ rho)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS)
    return rho.value

def partial_transpose(rho, dims, subsystem='B'):
    # Return the partial transpose of density matrix rho on the given subsystem.
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
        print(f"Warning: rho is not Hermitian (max diff = {max_diff:.3e}).")
        valid = False
    elif verbose:
        print("rho is Hermitian.")
    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) < -tol:
        print(f"Warning: rho is not PSD (min eigenvalue = {np.min(eigvals):.3e}).")
        valid = False
    elif verbose:
        print(f"rho is PSD (min eigenvalue = {np.min(eigvals):.3e}).")
    trace_val = np.trace(rho).real
    if abs(trace_val - 1.0) > tol:
        print(f"Warning: Tr(rho) = {trace_val:.6f} (should be 1).")
        valid = False
    elif verbose:
        print(f"Tr(rho) = {trace_val:.6f}.")
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

def _single_restart(args):
    d, inter = args
    # Sort POVMs 
    # Alice has d two outcomes measurements (X = d, A = 2)
    # Bob has one d-outcome and one two-outcome measurements
    alice_povm = random_POVM_partie(d, 2, d)
    bob_povm = random_POVM_partie(1, d, d) + random_POVM_partie(1, 2, d)
    # Construct G and optimize rho
    G = construct_G(alice_povm, bob_povm, d)
    rho = optimize_rho_TB(G, d)
    # Calculate the initial result and initialize tracking variables
    maximal_result = -10
    result = -10
    maximal_alice = alice_povm
    maximal_bob = bob_povm
    # Loop until convergence
    count = 0
    results = []
    while True:
        alice_povm = SDP_alice(bob_povm, rho, d)
        bob_povm = SDP_bob(alice_povm, rho, d)
        G = construct_G(alice_povm, bob_povm, d)
        rho = optimize_rho_TB(G, d)
        previous_result = result
        result = np.trace(rho @ G)
        results.append(result.real)
        if abs(result.real - previous_result) / abs(previous_result) <= 1e-4 or count >= 50:
            if result >= maximal_result:
                maximal_result = result
                maximal_bob = bob_povm
                maximal_alice = alice_povm
            break
        count += 1
        if result >= maximal_result:
            maximal_result = result
            maximal_bob = bob_povm
            maximal_alice = alice_povm
    return maximal_result, maximal_alice, maximal_bob

def SeeSaw_PPT_family(d, n=1000, max_workers=None):
    global_max = -10
    global_alice = None
    global_bob = None
    with fut.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for maximal_result, maximal_alice, maximal_bob in pool.map(_single_restart, [(d, inter) for inter in range(n)]):
            if maximal_result > global_max:
                global_max = maximal_result
                global_alice = maximal_alice
                global_bob = maximal_bob
    return global_max, global_alice, global_bob

def test_seesaw_algorithm(d, n=1000, max_workers=None, max_iter=15, tol=1e-6, verbose=False):
    print("Running SeeSaw algorithm...")
    global_max, alice_POVM, bob_POVM = SeeSaw_PPT_family(d, n, max_workers)
    G = construct_G(alice_POVM, bob_POVM, d)
    rho = optimize_rho_TB(G, d)
    print("\n--- Validity Checks ---")
    print("Checking density matrix rho...")
    dm_valid = check_density_matrix(rho, verbose=verbose, tol=tol)
    print("Checking PPT condition on rho...")
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

if __name__ == "__main__":
    d = int(sys.argv[1])
    n = int(sys.argv[2])
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(test_seesaw_algorithm(d, n, max_workers))
