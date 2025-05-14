#!/usr/bin/env python3
import os
import csv
import glob
import numpy as np
import pandas as pd
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
    inv_sqrt_S = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.conj().T
    return [inv_sqrt_S @ W @ inv_sqrt_S for W in Ws]

def random_POVM_partie(n_settings, n_outcomes, d):
    """
    For a given party (Alice or Bob), generate a list of POVMs (one per measurement setting)
    using the random_povm function.
    """
    return [random_povm(n_outcomes, d) for _ in range(n_settings)]

def construc_G(inequality_line, POVM_alice, POVM_bob, scenario_tuple, d):
    """
    Construct the operator G from the provided inequality coefficients and
    the POVMs of Alice and Bob. The coefficients are split into joint and marginal terms.
    """
    X, Y, A, B = scenario_tuple
    numbers = list(map(float, inequality_line.strip().split()))
    n_joint = (A-1)*X*(B-1)*Y
    n_alice = (A-1)*X
    n_bob   = (B-1)*Y
    if len(numbers) != n_joint + n_alice + n_bob:
        raise Exception(f"Expected {n_joint+n_alice+n_bob} coefficients, got {len(numbers)}.")

    joint_prob      = numbers[:n_joint]
    alice_marginal  = numbers[n_joint:n_joint+n_alice]
    bob_marginal    = numbers[n_joint+n_alice:]

    G = 0
    idx = 0
    # Alice marginal terms
    for a in range(A-1):
        for x in range(X):
            G += alice_marginal[idx] * np.kron(POVM_alice[x][a], np.eye(d))
            idx += 1
    idx = 0
    # Bob marginal terms
    for b in range(B-1):
        for y in range(Y):
            G += bob_marginal[idx] * np.kron(np.eye(d), POVM_bob[y][b])
            idx += 1
    idx = 0
    # Joint terms
    for a in range(A-1):
        for b in range(B-1):
            for x in range(X):
                for y in range(Y):
                    G += joint_prob[idx] * np.kron(POVM_alice[x][a], POVM_bob[y][b])
                    idx += 1
    return G

def optimize_rho_TB(G, d):
    """
    Optimize the bipartite state rho to maximize Tr(G rho) subject to:
      - rho is positive semidefinite.
      - rho has unit trace.
      - rho satisfies the PPT condition (partial transpose on Bob's subsystem).
    """
    d_tot = d**2
    rho   = cp.Variable((d_tot, d_tot), hermitian=True)
    sigma = cp.Variable((d_tot, d_tot), hermitian=True)

    constraints = [
        rho >> 0,
        sigma >> 0,
        cp.trace(rho) == 1
    ]
    # PPT constraint: sigma == partial_transpose(rho)
    for iA in range(d):
        for iB in range(d):
            for jA in range(d):
                for jB in range(d):
                    r  = iA*d + iB
                    c  = jA*d + jB
                    pr = iA*d + jB
                    pc = jA*d + iB
                    constraints.append(sigma[pr, pc] == rho[r, c])

    objective = cp.Maximize(cp.real(cp.trace(G @ rho)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT, warm_start=False)
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
    Optimize Alice's POVM elements while keeping Bob's fixed.
    """
    X, Y, A, B = scenario_tuple
    alice_POVM = [[cp.Variable((d, d), hermitian=True) for _ in range(A)] for _ in range(X)]

    numbers = list(map(float, inequality_line.strip().split()))
    n_joint = (A-1)*X*(B-1)*Y
    n_alice = (A-1)*X
    if len(numbers) != n_joint + n_alice + (B-1)*Y:
        raise Exception("Coefficient mismatch in SDP_alice.")
    joint_prob     = numbers[:n_joint]
    alice_marginal = numbers[n_joint:n_joint+n_alice]

    G = 0
    idx = 0
    # Alice marginal contributions
    for a in range(A-1):
        for x in range(X):
            G += alice_marginal[idx] * general_kron(alice_POVM[x][a], np.eye(d))
            idx += 1
    idx = 0
    # Joint contributions
    for a in range(A-1):
        for b in range(B-1):
            for x in range(X):
                for y in range(Y):
                    G += joint_prob[idx] * general_kron(alice_POVM[x][a], bob_POVM[y][b])
                    idx += 1

    objective = cp.Maximize(cp.real(cp.trace(G @ rho)))
    constraints = []
    for setting in alice_POVM:
        for E in setting:
            constraints.append(E >> 0)
        constraints.append(sum(setting) == np.eye(d))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT, warm_start=False)

    return [[E.value for E in setting] for setting in alice_POVM]

def SDP_bob(alice_POVM, rho, inequality_line, scenario_tuple, d):
    """
    Optimize Bob's POVM elements while keeping Alice's fixed.
    """
    X, Y, A, B = scenario_tuple
    bob_POVM = [[cp.Variable((d, d), hermitian=True) for _ in range(B)] for _ in range(Y)]

    numbers = list(map(float, inequality_line.strip().split()))
    n_joint = (A-1)*X*(B-1)*Y
    n_alice = (A-1)*X
    if len(numbers) != n_joint + n_alice + (B-1)*Y:
        raise Exception("Coefficient mismatch in SDP_bob.")
    joint_prob    = numbers[:n_joint]
    bob_marginal  = numbers[n_joint+n_alice:]

    G = 0
    idx = 0
    # Bob marginal contributions
    for b in range(B-1):
        for y in range(Y):
            G += bob_marginal[idx] * general_kron(np.eye(d), bob_POVM[y][b])
            idx += 1
    idx = 0
    # Joint contributions
    for a in range(A-1):
        for b in range(B-1):
            for x in range(X):
                for y in range(Y):
                    G += joint_prob[idx] * general_kron(alice_POVM[x][a], bob_POVM[y][b])
                    idx += 1

    objective = cp.Maximize(cp.real(cp.trace(G @ rho)))
    constraints = []
    for setting in bob_POVM:
        for F in setting:
            constraints.append(F >> 0)
        constraints.append(sum(setting) == np.eye(d))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT, warm_start=True)

    return [[F.value for F in setting] for setting in bob_POVM]

def SeeSaw(scenario, inequality_line, d, n=1000, max_iter=15, tol=1e-9):
    """
    Perform the See-Saw iterative algorithm:
      1. Randomly initialize POVMs for Alice and Bob.
      2. Optimize the state rho under PPT constraints.
      3. Alternately optimize Alice's and Bob's POVMs until convergence.
      4. Track the maximal violation over multiple initializations.
    """
    X, Y, A, B = scenario
    global_max   = -np.inf
    global_alice = None
    global_bob   = None

    for init in range(n):
        alice_povm = random_POVM_partie(X, A, d)
        bob_povm   = random_POVM_partie(Y, B, d)
        G = construc_G(inequality_line, alice_povm, bob_povm, scenario, d)
        rho = optimize_rho_TB(G, d)
        best_val = np.real(np.trace(rho @ G))
        prev_val = best_val
        count    = 0

        while True:
            alice_povm = SDP_alice(bob_povm, rho, inequality_line, scenario, d)
            bob_povm   = SDP_bob(alice_povm, rho, inequality_line, scenario, d)
            G = construc_G(inequality_line, alice_povm, bob_povm, scenario, d)
            rho = optimize_rho_TB(G, d)
            curr_val = np.real(np.trace(rho @ G))

            if curr_val > best_val:
                best_val = curr_val
                best_alice, best_bob = alice_povm, bob_povm

            if count >= max_iter or abs(curr_val - prev_val)/max(abs(curr_val),1) <= tol:
                break

            prev_val = curr_val
            count += 1

        logging.info(f"Init {init}: best={best_val:.6f}")
        if best_val > global_max:
            global_max   = best_val
            global_alice = best_alice
            global_bob   = best_bob

    return global_max, global_alice, global_bob

# -------------------- Main Execution -------------------- #

if __name__ == "__main__":
    d = 3  # Hilbert space dimension
    col_name    = f"see-saw_{d}"
    base_file   = "consolidated_results.csv"
    target_file = "see-saw_results.csv"

    # Create or update the results CSV
    if not os.path.exists(target_file):
        df = pd.read_csv(base_file)
        df[col_name] = 0.0
        df.to_csv(target_file, index=False)
    else:
        df = pd.read_csv(target_file)
        if col_name not in df.columns:
            df[col_name] = 0.0
            df.to_csv(target_file, index=False)

    # Select the next 5 pending inequalities
    pending = df[df[col_name] == 0.0].head(5)
    if pending.empty:
        print("No pending inequalities for See-Saw.")
        exit(0)

    # Detailed log file
    log_file = f"seesaw_results_d{d}.txt"
    with open(log_file, "a") as f:
        for idx, row in pending.iterrows():
            scenario = tuple(map(int, row["scenario"]))
            ineq     = row["inequality"]
            f.write("="*40 + "\n")
            f.write(f"Scenario: {row['scenario']}\n")
            f.write(f"Inequality: {ineq}\n")
            print(f"Running SeeSaw for scenario {row['scenario']}")

            # Run See-Saw
            val, alice_POVM, bob_POVM = SeeSaw(scenario, ineq, d)

            # Record results
            f.write(f"Final Violation: {val}\n")
            f.write("Alice POVMs:\n")
            for x, setting in enumerate(alice_POVM):
                for a, E in enumerate(setting):
                    f.write(f"  Setting {x}, Outcome {a}:\n")
                    f.write(np.array2string(E, precision=3) + "\n")
            f.write("Bob POVMs:\n")
            for y, setting in enumerate(bob_POVM):
                for b, F in enumerate(setting):
                    f.write(f"  Setting {y}, Outcome {b}:\n")
                    f.write(np.array2string(F, precision=3) + "\n")
            f.write("\n")

            # Update CSV
            df.at[idx, col_name] = val
            df.to_csv(target_file, index=False)
            print(f"Done line {idx}, violation={val}")

    print(f"See-Saw results logged to {log_file}.")
