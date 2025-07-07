import numpy as np
import qutip
import cvxpy as cp
import sys

def random_POVM_partie(n_settings, n_outcomes, d=4):
    #Sort random a POVM for one partie and return as a list such that: POVM[x][a] = M_{a|x}
    POVM = []
    for _ in range(n_settings):
        POVM_i = []
        for _ in range(n_outcomes-1):

            POVM_i.append(qutip.rand_dm(d).full())
        
        POVM_i.append(np.identity(d) - sum(POVM_i))



        POVM.append(POVM_i)
    
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
        povm_two_outcome.append(bob_POVM[0][b].value)
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

if __name__ == "__main__":
    SeeSaw_PPT_family(sys.argv[1],sys.argv[2])