
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
    


def construct_bell_operator(alice_povm,bob_povm,d):
    G = (d-2)*(np.kron(alice_povm[0][0],bob_povm[1][0] - bob_povm[0][0]))
    for i in range(1,d):
        for j in range(1,d):
            if i!=j:
                G += -np.kron(alice_povm[i][0],bob_povm[0][j])
    for i in range(1,d):
        G += - np.kron(alice_povm[i][1],bob_povm[1][0])
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


def SDP_alice(rho,bob_povm,d):
    # Define variables to be optimized
    X = d
    A = 2
    alice_POVM = []
    for _ in range(X):
        alice_POVM_x = []
        for _ in range(A):
            alice_POVM_x.append(cp.Variable((d, d), symmetric=True))
        alice_POVM.append(alice_POVM_x)

    #Define objective function
    

    rho_00 = -(d-2)*partial_trace_numpy(rho@(np.kron(np.identity(d), bob_povm[0][0])),d,"B")
    rho_00 += (d-2)*partial_trace_numpy(rho@(np.kron(np.identity(d), bob_povm[1][0])),d,"B")
    
    rho_0i = [0 for i in range(d)]
    rho_1i = [0 for i in range(d)]
    for i in range(1,d):
        rho_1i[i] += -1*partial_trace_numpy(rho@(np.kron(np.identity(d), bob_povm[1][0])),d,"B")
        for j in range(1,d):
            if i !=j:
                rho_0i[i] += -1*partial_trace_numpy(rho@(np.kron(np.identity(d), bob_povm[0][j])),d,"B")

    expr = 0
    for x,setting in enumerate(alice_POVM):
        for a,measurement in enumerate(setting):
            if a == 0 and x ==0:
                expr += cp.trace(rho_00@measurement)
            elif a == 0 and (d-1) >= x >=1 :
                expr += cp.trace(rho_0i[x]@measurement)
            elif a == 1 and x!=0:
                expr += cp.trace(rho_1i[x]@measurement)
    
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
        POVM_constraints.append(S == np.identity(d))
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



def SDP_bob(rho, alice_povm,d):
    # Define variables to be optimized

    bob_POVM = []
    bob_POVM_d_outcome = []
    for _ in range(d):
        bob_POVM_d_outcome.append(cp.Variable((d, d), symmetric=True))
    bob_POVM.append(bob_POVM_d_outcome)
    bob_POVM_two_outcome = []
    for _ in range(2):
        bob_POVM_two_outcome.append(cp.Variable((d, d), symmetric=True))
    bob_POVM.append(bob_POVM_two_outcome)

    #Define objective function
    

    rho_00 = -(d-2)*partial_trace_numpy(rho@np.kron(alice_povm[0][0],np.identity(d)),d,"A")
    rho_01 = (d-2)*partial_trace_numpy(rho@np.kron(alice_povm[0][0],np.identity(d)),d,"A")
    rho_j0 = [0 for j in range(d)]
    for i in range(1,d):
        rho_01 += -1*partial_trace_numpy(rho@np.kron(alice_povm[i][1],np.identity(d)),d,"A")
        for j in range(1,d):
            if i !=j:
                rho_j0[j] += -1*partial_trace_numpy(rho@np.kron(alice_povm[i][0],np.identity(d)),d,"A") 
    
    expr = 0
    for y,setting in enumerate(bob_POVM):
        for b,measurement in enumerate(setting):
            if b == 0 and y ==0:
                expr += cp.trace(rho_00@measurement)
            elif b == 0 and y == 1:
                expr += cp.trace(rho_01@measurement)
            elif b != 0 and y == 0:
                expr += cp.trace(rho_j0[b]@measurement)
    
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
        POVM_constraints.append(S  == np.identity(d))
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

# This function optimizes rho s.t. it is PPT
def optimize_rho_TB(G, d):
    d_total = d ** 2
    rho = cp.Variable((d_total, d_total), symmetric=True)
    rho_pt = cp.partial_transpose(rho, dims=[d, d], axis=1)
    constraints = [rho >> 0, 	rho_pt >> 0, cp.trace(rho) == 1]
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








