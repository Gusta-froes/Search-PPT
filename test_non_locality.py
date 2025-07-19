from datetime import datetime
import numpy as np
import re
import multiprocessing as mp
import gurobipy as gp
from gurobipy import GRB
from itertools import product
import argparse

def probs(M, N, rho, clip_tol=0):
    n_x = len(M)
    n_y = len(N)
    n_a = len(M[0])
    n_b = len(N[0])
    T = n_a * n_b * n_x * n_y
    p = np.empty(T, dtype=float)
    for x in range(n_x):
        for y in range(n_y):
            block_indices = []
            block_vals = []
            for a, Mxa in enumerate(M[x]):
                for b, Nyb in enumerate(N[y]):
                    k = a + n_a*b + n_a*n_b*x + n_a*n_b*n_x*y
                    val = np.trace(np.kron(Mxa, Nyb) @ rho).real
                    p[k] = val
                    if clip_tol > 0:
                        block_indices.append(k)
                        block_vals.append(val)
            if clip_tol > 0:
                block_vals = np.array(block_vals)
                block_vals[block_vals < -clip_tol] = 0.0
                block_vals[(block_vals >= -clip_tol) & (block_vals < 0)] = 0.0
                s = block_vals.sum()
                if s > 0:
                    block_vals /= s
                for bi, vval in zip(block_indices, block_vals):
                    p[bi] = vval
    return p

def deterministic_vertices(n_x, n_y, n_a, n_b, as_array=True):
    T = n_a * n_b * n_x * n_y
    def gen():
        for fA in product(range(n_a), repeat=n_x):
            for fB in product(range(n_b), repeat=n_y):
                v = np.zeros(T, dtype=float)
                for x, a in enumerate(fA):
                    for y, b in enumerate(fB):
                        k = a + n_a*b + n_a*n_b*x + n_a*n_b*n_x*y
                        v[k] = 1.0
                yield v
    if as_array:
        return np.vstack(list(gen()))
    else:
        return gen()

def init_worker(As, Bs, D, rhos):
    global As_global, Bs_global, D_global, rhos_global
    global model_global, constrs_global, q_global, t_global
    As_global, Bs_global, D_global, rhos_global = As, Bs, D, rhos
    m, T = D_global.shape
    model_global = gp.Model()
    model_global.setParam('OutputFlag', 0)
    model_global.setParam('Threads', 1)
    q_global = model_global.addVars(m, lb=-GRB.INFINITY, name="q")
    t_global = model_global.addVar(lb=-GRB.INFINITY, name="t")
    model_global.setObjective(t_global, GRB.MAXIMIZE)
    model_global.addConstr(q_global.sum() == 1)
    for i in range(m):
        model_global.addConstr(q_global[i] >= t_global)
    A_mat = D_global.T
    q_list = [q_global[i] for i in range(m)]
    constrs_global = model_global.addMConstr(A_mat, q_list, '=', np.zeros(T))
    model_global.update()

def worker(idx_state):
    rho = rhos_global[idx_state]
    failures = []
    for meas_idx, (A, B) in enumerate(zip(As_global, Bs_global)):
        p_vec = probs(A, B, rho)
        for constr, val in zip(constrs_global, p_vec):
            constr.rhs = val
        model_global.optimize()
        if model_global.status != GRB.OPTIMAL or t_global.X < 0:
            failures.append((idx_state, meas_idx))
    return failures


def batch_test_locality_parallel_optimized(rho_file, meas_files, d, n_workers, N=None, n_measurements=None):
    global rhos_global
    rhos = np.load(rho_file, allow_pickle=True)
    if N is not None:
        rhos = rhos[:N]
    rhos_global = rhos
    start_time = datetime.now()
    log = open(f"log_started_{start_time.strftime('%Y%m%d_%H%M%S')}_d_{d}.txt", 'w')
    for meas_file in meas_files:
        m = re.search(r"x(\d+)_y(\d+)_a(\d+)_b(\d+)", meas_file)
        if not m:
            continue
        n_inputs_a, n_inputs_b, n_outputs_a, n_outputs_b = map(int, m.groups())
        data = np.load(meas_file, allow_pickle=True)
        As = data['A']
        Bs = data['B']
        use_n = len(As) if n_measurements is None else min(n_measurements, len(As))
        As, Bs = As[:use_n], Bs[:use_n]
        D = deterministic_vertices(n_inputs_a, n_inputs_b, n_outputs_a, n_outputs_b)
        print(f"[{datetime.now():%H:%M:%S}] Doing configuration: x={n_inputs_a}, y={n_inputs_b}, a={n_outputs_a}, b={n_outputs_b}")
        args = list(range(len(rhos)))
        with mp.Pool(processes=n_workers,
                    initializer=init_worker,
                    initargs=(As, Bs, D, rhos_global)) as pool:
            results = pool.map(worker, list(range(len(rhos_global))))
        for failures_list in results:
            for state_idx, meas_idx in failures_list:
                ts = datetime.now().strftime("%H:%M:%S")
                line = f"[{ts}] State {state_idx}, measurement {meas_idx} not local for config (x={n_inputs_a}, y={n_inputs_b}, a={n_outputs_a}, b={n_outputs_b}), file {meas_file}"
                print(line)
                log.write(line + "\n")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho_file", required=True)
    parser.add_argument("--meas_files", nargs="+", required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--n_measurements", type=int, default=None)
    args = parser.parse_args()
    batch_test_locality_parallel_optimized(
        args.rho_file,
        args.meas_files,
        args.d,
        args.n_workers,
        args.N,
        args.n_measurements
    )
