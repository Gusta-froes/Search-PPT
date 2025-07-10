
import numpy as np
import qutip
import cvxpy as cp
import sys
from toqito.rand import random_povm
import concurrent.futures as fut

import multiprocessing as mp



def random_POVM_partie(n_settings, n_outcomes, d=4):
    #Sort random a POVM for one partie and return as a list such that: POVM[x][a] = M_{a|x}
    POVM = []
    povm_ndarray = random_povm(d, n_settings, n_outcomes)

    POVM = [
    [ povm_ndarray[:, :, x, a].real for a in range(n_outcomes) ]
    for x in range(n_settings)
]
    
    return POVM

def isPSD(A, tol=1e-8):
  E = np.linalg.eigvalsh(A)
  return np.all(E > -tol)


d = 3
for _ in range(1000):
    povm = random_POVM_partie(d,2,d)
    for setting in povm:
        S = 0
        for measurement in setting:
            S += measurement
            assert isPSD(measurement)
        assert np.allclose(S,np.identity(d))
    


def construct_bell_operator(alice_povm, bob_povm, d):
    G = np.zeros((d**2, d**2))
    G += (d-2) * (np.kron(alice_povm[0][0], bob_povm[1][0] - bob_povm[0][0]))
    
    for i in range(1, d):
        for j in range(1, d):
            if i != j:
                G -= np.kron(alice_povm[i][0], bob_povm[0][j])
                
    for i in range(1, d):
        G -= np.kron(alice_povm[i][1], bob_povm[1][0])
        
    return G



def partial_trace_numpy(rho, d, subsystem='B'):
    rho4 = rho.reshape(d, d, d, d)          
    if subsystem.upper() == 'A':
        return np.trace(rho4, axis1=0, axis2=2)   # trace A
    elif subsystem.upper() == 'B':
        return np.trace(rho4, axis1=1, axis2=3)   # trace B
    else:
        raise ValueError("subsystem must be 'A' or 'B'")


MOSEK_PARAMS = {
    "MSK_IPAR_NUM_THREADS": 1,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10
}


def SDP_alice(rho, bob_povm, d):
    # Precompute marginal operators
    rho_A_B00 = partial_trace_numpy(rho @ np.kron(np.eye(d), bob_povm[0][0]), d, "B")
    rho_A_B01 = partial_trace_numpy(rho @ np.kron(np.eye(d), bob_povm[1][0]), d, "B")
    rho_A_Bj0 = [None] * d
    for j in range(1, d):
        rho_A_Bj0[j] = partial_trace_numpy(rho @ np.kron(np.eye(d), bob_povm[0][j]), d, "B")

    # Initialize F_i matrices
    F = [None] * d
    F[0] = (d-2) * (rho_A_B01 - rho_A_B00)
    for i in range(1, d):
        F[i] = -sum(rho_A_Bj0[j] for j in range(1, d) if j != i) + rho_A_B01

    # SDP setup
    alice_povm_vars = []
    for _ in range(d):
        A0 = cp.Variable((d, d), symmetric=True)
        alice_povm_vars.append([A0, np.eye(d) - A0])  # A1 = I - A0

    obj = 0
    for i in range(d):
        obj += cp.trace(F[i] @ alice_povm_vars[i][0])
    
    constraints = [A0 >> 0 for i in range(d) for A0 in [alice_povm_vars[i][0]]] + \
                 [alice_povm_vars[i][0] << np.eye(d) for i in range(d)]
    
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS)
        
    return [[A0[0].value, np.eye(d) - A0[0].value] for A0 in alice_povm_vars]



def SDP_bob(rho, alice_povm, d):
    # Precompute marginal operators
    rho_B_A0i = [None] * d
    for i in range(d):
        rho_B_A0i[i] = partial_trace_numpy(rho @ np.kron(alice_povm[i][0], np.eye(d)), d, "A")
    
    rho_B_A1i = [None] * d
    for i in range(1, d):  # Note: i>=1
        rho_B_A1i[i] = partial_trace_numpy(rho @ np.kron(alice_povm[i][1], np.eye(d)), d, "A")

    # SDP setup
    B0 = [cp.Variable((d, d), symmetric=True) for _ in range(d)]  # y=0 outcomes
    C0 = cp.Variable((d, d), symmetric=True)  # y=1 outcome 0
    C1 = np.eye(d) - C0  # y=1 outcome 1

    # Objective terms
    obj = 0
    obj += (d-2) * cp.trace(rho_B_A0i[0].T @ (C0 - B0[0]))
    for i in range(1, d):
        for j in range(1, d):
            if i != j:
                obj -= cp.trace(rho_B_A0i[i].T @ B0[j])
    for i in range(1, d):
        obj -= cp.trace(rho_B_A1i[i].T @ C0)

    constraints = [var >> 0 for var in B0] + [sum(B0) == np.eye(d)] + [C0 >> 0, C1 >> 0]
    
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS)
    
    return [[B.value for B in B0], [C0.value, C1.value]]

# This function optimizes rho s.t. it is PPT
def optimize_rho_TB(G, d):
    d_total = d ** 2
    rho = cp.Variable((d_total, d_total), symmetric=True)
    rho_pt = cp.partial_transpose(rho, dims=[d, d], axis=1)
    constraints = [rho >> 0,rho_pt>>0 ,cp.trace(rho) == 1]
    objective = cp.Maximize(cp.trace(G @ rho))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS)
    return np.real_if_close(rho.value)





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
    if not np.allclose(rho, rho.T, atol=tol):
        max_diff = np.max(np.abs(rho - rho.T))
        print(f"Warning: rho is not symmetric (max diff = {max_diff:.3e}).")
        valid = False
    elif verbose:
        print("rho is symmetric.")
    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) < -tol:
        print(f"Warning: rho is not PSD (min eigenvalue = {np.min(eigvals):.3e}).")
        valid = False
    elif verbose:
        print(f"rho is PSD (min eigenvalue = {np.min(eigvals):.3e}).")
    trace_val = np.trace(rho)
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
        sum_op = np.zeros((dim, dim))
        for j, E in enumerate(elements):
            if not np.allclose(E, E.T, atol=tol):
                print(f"Warning: POVM set {idx}, element {j} not symmetric.")
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
    alice_povm = random_POVM_partie(d, 2, d)
    bob_povm = random_POVM_partie(1, d, d) + random_POVM_partie(1, 2, d)
    maximal_result = -10
    result = -10
    maximal_alice = alice_povm
    maximal_bob = bob_povm
    count = 0
    results = []
    while True:
        G = construct_bell_operator(alice_povm, bob_povm, d)
        rho = optimize_rho_TB(G, d)
        alice_povm = SDP_alice(rho,bob_povm, d)
        bob_povm = SDP_bob(rho,alice_povm, d)
        
        previous_result = result
        G = construct_bell_operator(alice_povm, bob_povm, d)

        result = np.trace(rho @ G)
        results.append(result)
        if abs(result - previous_result) / max(1e-12, abs(previous_result)) <= 1e-5 or count >= inter:
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
        for idx, (maximal_result, maximal_alice, maximal_bob) in enumerate(pool.map(_single_restart, [(d, inter) for inter in range(n)]), 1):
            if maximal_result > global_max:
                global_max = maximal_result
                global_alice = maximal_alice
                global_bob = maximal_bob
            if idx % 10 == 0:
                print(f"{idx}-iteration, Maximal violation so far: {global_max}")
    return global_max, global_alice, global_bob

def test_seesaw_algorithm(d, n=1000, max_workers=None, max_iter=15, tol=1e-6, verbose=False):
    print("Running SeeSaw algorithm...")
    global_max, alice_POVM, bob_POVM = SeeSaw_PPT_family(d, n, max_workers)
    G = construct_bell_operator(alice_POVM, bob_POVM, d)
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
    d = 3
    n_restarts = 100
    max_iter = 25
    max_workers = 1
    best_val, alice_POVM, bob_POVM = SeeSaw_PPT_family(d, n_restarts, max_workers)
    print("Best Bell violation:", best_val)
    rho_best = optimize_rho_TB(construct_bell_operator(alice_POVM, bob_POVM, d), d)
    print("ρ validity:", check_density_matrix(rho_best))








