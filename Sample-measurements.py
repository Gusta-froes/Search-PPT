import numpy as np
from itertools import combinations_with_replacement, product
import random 
import qutip


def partitions_fixed_M(N, M):
    for combo in combinations_with_replacement(range(0, N+1), M):
        if sum(combo) != N:
            continue
        num_pos = sum(1 for x in combo if x > 0)
        if num_pos >= 2:
            yield combo

def random_measurement(n_inputs,n_outputs,d):
    measurements = []
    for _ in range(n_inputs):
        measurement = []
        pointer = 0
        U = qutip.rand_unitary(d).full()
        partitions = [list(part) for part in partitions_fixed_M(d,n_outputs)]
        partition = sorted(random.choice(partitions),reverse=True)
        for outcome in range(n_outputs):
            p = partition[outcome]
            M = U@np.diag([0]*pointer+[1]*p +[0]*(d-(p+pointer)))@U.conj().T
            measurement.append(M)
            pointer += p
        measurements.append(measurement)
    return measurements

def probs(M, N, rho):
    n_x = len(M)
    n_y = len(N)
    P   = {}
    for x in range(n_x):
        for y in range(n_y):
            for a, Mxa in enumerate(M[x]):
                for b, Nyb in enumerate(N[y]):
                    P[(a,b,x,y)] = np.trace(np.kron(Mxa, Nyb)@ rho).real
    return P

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

def save_measurement_files(d, n_samples):
    for n_inputs_a in range(2, 2*d + 1):
        for n_inputs_b in range(2, 2*d + 1):
            for n_outputs_a in range(2, d + 1):
                for n_outputs_b in range(2, d + 1):
                    As = []
                    Bs = []
                    for _ in range(n_samples):
                        As.append(random_measurement(n_inputs_a, n_outputs_a, d))
                        Bs.append(random_measurement(n_inputs_b, n_outputs_b, d))
                    fname = f"measurements_d{d}_x{n_inputs_a}_y{n_inputs_b}_a{n_outputs_a}_b{n_outputs_b}.npz"
                    np.savez_compressed(fname, A=As, B=Bs)
                    print(f"Did scenario x-{n_inputs_a} y-{n_inputs_b} a-{n_outputs_a} b-{n_outputs_b}")



save_measurement_files(3,3000)