import numpy as np
import cvxpy as cp
import sys


def thetas_vertesi(d):
    assert d >= 3
    E = np.eye(d-1)
    mu = np.ones(d-1) / (d-1)
    norm_fact = np.sqrt((d-1)/(d-2))
    thetas = []
    for p in range(d-1):
        w = E[p] - mu
        vec = np.zeros(d)
        vec[1:] = norm_fact * w
        thetas.append(vec)
    return thetas
    
def generate_k_vectors(d):
    assert d >= 3
    E = np.eye(d-1)
    mu = np.ones(d-1) / (d-1)
    S_sub = (E - mu[np.newaxis, :]).T
    S = np.vstack([np.zeros((1, d-1)), S_sub])
    Q, _ = np.linalg.qr(S, mode='reduced')
    basis = Q[:, :d-2]
    return [basis[:, i] for i in range(d-2)]

def test_theta(theta, d, tol=1e-8):
    test = True
    arr = np.stack(theta)
    if not np.allclose(arr.sum(axis=0), 0, atol=tol):
        print("Failed the sum test")
        test = False

    for p in range(len(theta)):
        for q in range(len(theta)):
            ip = np.dot(theta[p], theta[q])
            if p == q:
                if not np.allclose(ip, 1, atol=tol):
                    print("Failed the norm test", p)
                    test = False
            else:
                if not np.allclose(ip, -1/(d-2), atol=tol):
                    print("Failed the inner product test", p, q)
                    test = False

    k_list = generate_k_vectors(d)
    alpha = (d-1)/(d-2)
    P_theta = sum(np.outer(v, v) for v in theta)
    P_k     = alpha * sum(np.outer(k, k) for k in k_list)
    if not np.allclose(P_theta, P_k, atol=tol):
        print("Failed Eq (12) test")
        test = False

    K_theta = sum(np.kron(v, v) for v in theta)
    K_k     = alpha * sum(np.kron(k, k) for k in k_list)
    if not np.allclose(K_theta.reshape((d, d)), K_k.reshape((d, d)), atol=tol):
        print("Failed Eq (13) test")
        test = False

    return test

for i in range(3,10):
    assert (test_theta(thetas_vertesi(i),i))

def generate_operators(theta, x, y, d):
    # Alice has d two-outcome measurements, a = d, x = 2
    # alice_povm[a][x] = A_{x|a}
    a = d
    alice_povm = [[None, None] for _ in range(d)]

    #Bob has one d-outcome and one two outcome 
    bob_povm = [[None]*d, [None, None]]

    e0 = np.zeros(d); e0[0] = 1
    e1 = np.zeros(d); e1[1] = 1

    for p in range(d):
        if p == 0:
            vec_A = e0
        else:
            vec_A = x[0]*e0 + x[1]*e1 + x[2]*theta[p-1]
        P0 = np.outer(vec_A, vec_A)
        alice_povm[p][0] = P0
        alice_povm[p][1] = np.eye(d) - P0

    for p in range(d):
        if p == 0:
            vec_B = -y[1]*e0 + y[0]*e1
        else:
            vec_B = (y[0]*e0 + y[1]*e1 + np.sqrt(d-2)*theta[p-1]) / np.sqrt(d-1)
        bob_povm[0][p] = np.outer(vec_B, vec_B)

    vec_B1 = e0
    P1 = np.outer(vec_B1, vec_B1)
    bob_povm[1][0] = P1
    bob_povm[1][1] = np.eye(d) - P1

    return alice_povm, bob_povm

# This function optimizes rho s.t. it is PPT
def optimize_rho_TB(G, d=4):

    d_total = d**2
    rho = cp.Variable((d_total, d_total), hermitian=True)
    rho_pt = cp.partial_transpose(rho, dims=[d, d], axis=1)


    constraints = [
        rho >> 0,             # rho is PSD
        rho_pt >> 0,           # PT_B(rho) is PSD
        cp.trace(rho) == 1    # Trace normalization 
    ]
    objective = cp.Maximize(cp.real(cp.trace(G @ rho)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 8})


    return rho.value

def construct_G(alice_povm, bob_povm, d):
    # Contruct the bell operator G from the family of inequalities described in https://arxiv.org/pdf/1704.08600
    G = (d-2)*(np.kron(alice_povm[0][0],bob_povm[1][0]) -  np.kron(alice_povm[0][0],bob_povm[0][0]))
    
    for i in range(1,d):
        
        G += -(np.kron(np.identity(d), bob_povm[1][0]) - np.kron(alice_povm[i][0], bob_povm[1][0]))

        for j in range(1,d):
            if i!=j:    
                G += -np.kron(alice_povm[i][0], bob_povm[0][j])
    return G

def brute_force_opt(d, grid_points=20, angle_points=20):
    theta = thetas_vertesi(d)
    best = {'value': -np.inf, 'x': None, 'y': None}
    grid = np.linspace(0, 1, grid_points)
    total = grid_points**2 * angle_points
    count = 0
    for x0 in grid:
        for x1 in grid:
            if x0**2 + x1**2 > 1:
                count += angle_points
                continue
            x2 = np.sqrt(1 - x0**2 - x1**2)
            x = [x0, x1, x2]
            for phi in np.linspace(0, 2 * np.pi, angle_points):
                y0 = np.cos(phi)
                y1 = np.sin(phi)
                y = [y0, y1]
                alice_povm, bob_povm = generate_operators(theta, x, y, d)
                G = construct_G(alice_povm, bob_povm, d)
                rho_opt = optimize_rho_TB(G, d)
                val = np.real(np.trace(G @ rho_opt))
                count += 1
                if val > best['value']:
                    best.update({'value': val, 'x': x, 'y': y})
                    print(f"[{count}/{total}] New best violation: {val:.6f} at x={x}, y={y}")
    return best

if __name__ == "__main__":
    d = sys.argv[1]
    x_points, y_points = sys.argv[2], sys.argv[3]
    brute_force_opt(d,x_points,y_points)
