import numpy as np
import qutip, sys, json
from pathlib import Path


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

def check_density_matrix(rho, tol=1e-10, verbose=False):
    valid = True
    if not np.allclose(rho, rho.T.conj(), atol=tol):
        max_diff = np.max(np.abs(rho - rho.conj().T))
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


def check_ppt_condition(rho, dims, tol=1e-10, verbose=False):
    rho_pt = partial_transpose(rho, dims, subsystem='B')
    eigenvals = np.linalg.eigvalsh(rho_pt)
    min_eig = np.min(eigenvals)
    valid = True
    if min_eig < -tol:
        print(f"Warning: PPT condition violated (min eigenvalue = {min_eig:.3e}).")
        valid =  False
    else:
        if verbose:
            print("PPT condition satisfied.")

        valid = True
    return valid 

def naive_pt(rho,d):
    eigenvalues, eigenvectors = np.linalg.eigh(partial_transpose(rho,[d,d]))

    new_rho = np.zeros_like(rho, dtype=complex)
    new_eigenvalues = eigenvalues.copy()
    for j, lam in enumerate(eigenvalues):
        if lam < 0:
            new_eigenvalues[j] = 0
    
    eigenvalues = [i/np.sum(new_eigenvalues) for i in new_eigenvalues]

    for j, lam in enumerate(eigenvalues):
        v = eigenvectors[:, j]            
        new_rho += lam * np.outer(v, v.conj()) 
    
    rho = partial_transpose(new_rho,[d,d])
    return rho

def random_rho_boundary_PPT(d):
    rho = qutip.rand_dm(d**2, distribution = "pure").full()
    rho = naive_pt(rho,d)
    if check_ppt_condition(rho,[d,d]) and check_density_matrix(rho):
        return rho
    else:
        print("Failed to generate random rho in the boundary")
        return None
    

def main():
    if len(sys.argv) < 3:
        print("usage: python script.py <d> <n> [outfile]")
        sys.exit(1)

    d = int(sys.argv[1])
    n = int(sys.argv[2])
    outfile = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    states = []
    while len(states) < n:
        rho = random_rho_boundary_PPT(d)
        if rho is not None:
            states.append(rho)
            
    if outfile:
        if outfile.exists():
            old = np.load(outfile, allow_pickle=False)
            all_states = np.concatenate([old, np.stack(states)])
        else:
            all_states = np.stack(states)
        np.save(outfile, all_states)
        print(f"file {outfile} now contains {len(all_states)} states")
    else:
        for k, r in enumerate(states, 1):
            print(f"state {k}:\n", r, "\n")

if __name__ == "__main__":
    main()