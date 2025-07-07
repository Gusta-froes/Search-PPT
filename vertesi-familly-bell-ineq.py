import numpy as np
import qutip
import cvxpy as cp
import sys
from toqito.rand import random_povm


def random_POVM_partie(n_settings, n_outcomes, d=4):
    #Sort random a POVM for one partie and return as a list such that: POVM[x][a] = M_{a|x}
    POVM = []
    povm_ndarray = random_povm(d, n_settings, n_outcomes)

    POVM = [
    [ povm_ndarray[:, :, a, x] for a in range(n_outcomes) ]
    for x in range(n_settings)
]
    
    return POVM



def general_kron(a, b):
    """
    Returns a CVXPY Expression representing the Kronecker product of a and b.
    
    At most one of "a" and "b" may be CVXPY Variable objects.
    
    :param a: 2D numpy ndarray, or a CVXPY Variable with a.ndim == 2
    :param b: 2D numpy ndarray, or a CVXPY Variable with b.ndim == 2
    """
    
    
    # np.kron does not work for a CVXPY variable as argument (weird dimensions problem), got this implementation from: https://github.com/cvxpy/cvxpy/issues/457
    
    expr = np.kron(a, b)
    num_rows = expr.shape[0]
    rows = [cp.hstack(expr[i,:]) for i in range(num_rows)]
    full_expr = cp.vstack(rows)
    return full_expr


def construct_G(alice_povm, bob_povm, d, general_kron_var = False):
    # Contruct the bell operator G from the family of inequalities described in https://arxiv.org/pdf/1704.08600
    # Uses the general kron function if we are dealing with an CVXPY experession
    if not general_kron_var:
        G = (d-2)*(np.kron(alice_povm[0][0],bob_povm[1][0]) -  np.kron(alice_povm[0][0],bob_povm[0][0]))
        for i in range(1,d):
            G += -np.kron(alice_povm[i][1], bob_povm[1][0])
            for j in range(1,d):
                if i!=j:    
                    G += -np.kron(alice_povm[i][0], bob_povm[0][j])
    else:
        G = (d-2)*(general_kron(alice_povm[0][0],bob_povm[1][0]) -  general_kron(alice_povm[0][0],bob_povm[0][0]))
        for i in range(1,d):
            G += -general_kron(alice_povm[i][1], bob_povm[1][0])
            for j in range(1,d):
                if i!=j:
                    G += -general_kron(alice_povm[i][0], bob_povm[0][j])
        
    return G

def SDP_alice (bob_POVM, rho,d = 4):
    # Define variables to be optimized
    X = d
    A = 2
    alice_POVM = []
    for _ in range(X):

        alice_POVM_x = []

        for _ in range(A):

            alice_POVM_x.append(cp.Variable((d,d),  hermitian=True))
        
        alice_POVM.append(alice_POVM_x)
    
    # Define objective function: 
    G = construct_G(alice_POVM,bob_POVM,d, general_kron_var = True)

    obj = cp.Maximize(cp.real(cp.trace(G@rho))) # We need the real part because the maximization cant be done with a complex variable (even if the imaginary part is zero)

    # Define the POVM constraint
    # First the positive semi definite constraint
    PSD_constraints = []
    for settings in alice_POVM:
        for measurement in settings:
            PSD_constraints.append(  measurement  >> 0  )
    
    # POVM constraint

    POVM_constraints = []
    for settings in alice_POVM:
        S = 0
        for measurement in settings:
            S += measurement

        POVM_constraints.append( S - np.identity(d) == 0)

    constraints = PSD_constraints + POVM_constraints

    # Define the problem:

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 8})  # Returns the optimal value.



    optimized_alice_POVM = []
    for x in range(X):
        povm_x = []
        for a in range(A):
            povm_x.append(alice_POVM[x][a].value)
        optimized_alice_POVM.append(povm_x)
    
    return optimized_alice_POVM


def SDP_bob (alice_POVM, rho,d = 4):

   


    # Define variables to be optimized

    bob_POVM = []
    
    # d-outcome setting:
    bob_POVM_d_outcome = []

    for _ in range(d):

        bob_POVM_d_outcome.append(cp.Variable((d,d),  hermitian=True))
    
    bob_POVM.append(bob_POVM_d_outcome)

    # two-outcome setting:

    bob_POVM_two_outcome = []

    for _ in range(2):

        bob_POVM_two_outcome.append(cp.Variable((d,d),  hermitian=True))
    
    bob_POVM.append(bob_POVM_two_outcome)


    # Define objective function:

    G = construct_G(alice_POVM, bob_POVM, d, general_kron_var = True)

    obj = cp.Maximize(cp.real(cp.trace(G@rho))) # We need the real part because the maximization cant be done with a complex variable (even if the imaginary part is zero)

    # Define the POVM constraint
    # First the positive semi definite constraint
    PSD_constraints = []
    for settings in bob_POVM:
        for measurement in settings:
            PSD_constraints.append(  measurement >> 0  )



    
    # POVM constraint

    POVM_constraints = []
    for settings in bob_POVM:
        S = 0
        for measurement in settings:
            S += measurement

        POVM_constraints.append( S - np.identity(d) == 0)
    
    

    constraints = POVM_constraints + PSD_constraints

    # Define the problem:

    prob = cp.Problem(obj, constraints)
    prob.solve(solver = cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 8})  # Returns the optimal value.


    optimized_bob_POVM = []
    # d-outcome setting unpacking
    povm_d_outcome = []
    for b in range(d):
        povm_d_outcome.append(bob_POVM[0][b].value)
    optimized_bob_POVM.append(povm_d_outcome)
    
    # two-outcome setting unpacking
    povm_two_outcome = []
    for b in range(2):
        povm_two_outcome.append(bob_POVM[1][b].value)
    optimized_bob_POVM.append(povm_two_outcome)

    return optimized_bob_POVM



# This function optimizes rho s.t. it is PPT
def optimize_rho_TB(G, d=4):

    d_total = d**2
    rho = cp.Variable((d_total, d_total), hermitian=True)
    sigma = cp.Variable((d_total, d_total), hermitian=True)  # Will represent rho^(T_B)

    constraints = [
        rho >> 0,             # rho is PSD
        sigma >> 0,           # PT_B(rho) is PSD
        cp.trace(rho) == 1    # Trace normalization
    ]

    # Impose sigma = PT_B(rho): swap subsystem B indices in the matrix representation.
    for iA in range(d):
        for iB in range(d):
            for jA in range(d):
                for jB in range(d):
                    # Original indices in rho corresponding to |iA,iB><jA,jB|
                    orig_row = iA * d + iB
                    orig_col = jA * d + jB

                    # For T_B: swap the B indices between bra and ket.
                    # New row corresponds to |iA,jB> and new col to |jA,iB>.
                    pt_row = iA * d + jB
                    pt_col = jA * d + iB

                    constraints.append(sigma[pt_row, pt_col] == rho[orig_row, orig_col])

    objective = cp.Maximize(cp.real(cp.trace(G @ rho)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 8})


    return rho.value


def partial_transpose(rho, dims, subsystem='B'):
    """
    Return the partial transpose of density matrix rho on the given subsystem.
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



def SeeSaw_PPT_family(d, n=1000):
    global_max = -10
    for inter in range(n):
        # Sort POVMs 
        # Alice has d two outcomes measurements (X = d, A = 2)
        # Bob has one d-outcome and one two-outcome measurements

        alice_povm = random_POVM_partie(d, 2, d)

        bob_povm = random_POVM_partie(1, d, d) + random_POVM_partie(1,2,d)

        # Construct G and optimize rho
        G = construct_G(alice_povm, bob_povm, d)
        rho = optimize_rho_TB(G, d)

        # Calculate the initial result and initialize tracking variables
        maximal_result = np.trace(rho @ G)
        maximal_alice = alice_povm
        maximal_bob = bob_povm

        # Loop until convergence
        count = 0
        while True:
            alice_povm = SDP_alice(bob_povm, rho, d)
            bob_povm = SDP_bob(alice_povm, rho, d)
            G = construct_G(alice_povm, bob_povm, d)
            rho = optimize_rho_TB(G, d)
            result = np.trace(rho @ G)


            if result >= maximal_result:
                maximal_result = result
                maximal_bob = bob_povm
                maximal_alice = alice_povm

            if count >= 5:
                break
            count += 1

        if maximal_result > global_max:
            global_max = maximal_result
            global_bob = maximal_bob
            global_alice = maximal_alice

        #print for saninty
        if inter%10 == 0:
            print(f"{inter}-interation, Maximal violation found: {global_max}, current value: {maximal_result} ")


        

    return global_max, global_alice, global_bob


def test_seesaw_algorithm( d, n=1000, max_iter=15, tol=1e-6, verbose=False):
    print("Running SeeSaw algorithm...")
    global_max, alice_POVM, bob_POVM = SeeSaw_PPT_family( d, n)
    G = construct_G(alice_POVM, bob_POVM,d)
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
    print(test_seesaw_algorithm(int(sys.argv[1]),int(sys.argv[2])))