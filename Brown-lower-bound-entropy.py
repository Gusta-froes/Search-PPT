
import numpy as np
from math import sqrt, log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy


# Functions from Peter's code 

def objective(ti):
    """
    Returns the objective function for the faster computations.
    Randomness generation on X=0 and only two outcomes for Alice.

        ti  --    i-th node
    """
    obj = 0.0
    F = [A[0][0], 1-A[0][0]]
    for a in range(len(F)):
        obj += F[a] * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

    return obj

def score_constraints(score,d, YO = True):
    """
    Returns the bell ineq score constraint
    """
    # Yu and Oh
    if YO:
        G = A[0][0]*B[1][0]
        for p in range(d):
            if not p==0:
                G += -((1 - A[p][0])*B[1][0])
            if p == (d-1):
                G += -(A[p][0]*(1 - np.sum([B[0][i] for i in range(d-1)])))
            else:
                G += -A[p][0]*B[0][p]
    # Pal & Vertesi family
    else:
        G = (d-2)*(A[0][0]*(B[1][0] - B[0][0]))
        for i in range(1,d):
            G += -(1 - A[i][0])*B[i][0]
            for j in range(1,d):
                if i!=j:
                    if j != (d-1):
                        G += -A[i][0]*B[0][j]
                    else:
                        G += -A[i][0]*(1 - np.sum([B[0][p] for p in range(d-1)]))
    

    bell_expr = G
    return [bell_expr - score]


def get_subs():
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A,B))

    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([A,B]):
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})

    return subs

def get_extra_monomials():
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []

    # Add ABZ
    ZZ = Z + [Dagger(z) for z in Z]
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for z in ZZ:
                monos += [a*b*z]

    # Add monos appearing in objective function
    for z in Z:
        monos += [A[0][0]*Dagger(z)*z]

    return monos[:]


def generate_quadrature(m):
    """
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

        m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w

def compute_entropy(SDP):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

        SDP -- sdp relaxation object
    """
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems
    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k])

        SDP.set_objective(new_objective)
        SDP.solve('mosek', mosek_params=MOSEK_PARAMS)

        if SDP.status == 'optimal':
            # 1 contributes to the constant term
            ent += ck * (1 + SDP.dual)
        else:
            # If we didn't solve the SDP well enough then just bound the entropy
            # trivially
            ent = 0
            if VERBOSE:
                print('Bad solve: ', k, SDP.status)
            break

    return ent

LEVEL = 2                        # NPA relaxation level
M = 12                            # Number of nodes / 2 in gaussian quadrature
T, W = generate_quadrature(M)    # Nodes, weights of quadrature
KEEP_M = 0                        # Optimizing mth objective function?
VERBOSE = 1                        # If > 1 then ncpol2sdpa will also be verbose

MOSEK_PARAMS = {
    "MSK_IPAR_NUM_THREADS": 1,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10
}

# Yu and Oh 
test_score_YO = [0.0002652642115091151,7.084902550288176e-05,2.6146735005708298e-05,1.1769192155722503e-05,6.0509366591464105e-06,3.4208123704924778e-06,2.0767039670695177e-06]
# Pal and vertesi
test_score_PV = [0.000265264,0.000210913, 0.000162725,  0.000128375, 0.000103852,0.000085873]

for d in range(3,10):

    # Description of Alice and Bobs devices (each input has 2 outputs)
    A_config = [2 for _ in range(d)]
    B_config = [d,2]

    # Operators in the problem Alice, Bob and Eve
    A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
    B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
    Z = ncp.generate_operators('Z', 2, hermitian=0)

    substitutions = {}            # substitutions to be made (e.g. projections)
    moment_ineqs = []            # Moment inequalities (e.g. Tr[rho CHSH] >= c)
    moment_eqs = []                # Moment equalities (not needed here)
    op_eqs = []                    # Operator equalities (not needed here)
    op_ineqs = []                # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
    extra_monos = []            # Extra monomials to add to the relaxation beyond the level.


    # Get the relevant substitutions
    substitutions = get_subs()

    # Define the moment inequality related to chsh score
    score_cons = score_constraints(test_score_YO[d-3],d)

    # Get any extra monomials we wanted to add to the problem
    extra_monos = get_extra_monomials()

    # Define the objective function (changed later)
    obj = objective(1)

    # Finally defining the sdp relaxation in ncpol2sdpa
    ops = ncp.flatten([A,B,Z])
    sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
    sdp.get_relaxation(level = LEVEL,
        equalities = op_eqs[:],
        inequalities = op_ineqs[:],
        momentequalities = moment_eqs[:],
        momentinequalities = moment_ineqs[:] + score_cons[:],
        objective = obj,
        substitutions = substitutions,
        extramonomials = extra_monos)

    # # Test
    ent = compute_entropy(sdp)
    print(f"Yu and OH family SDP bound for d = {d}: {ent}")

for d in range(3,9):

    # Description of Alice and Bobs devices (each input has 2 outputs)
    A_config = [2 for _ in range(d)]
    B_config = [d,2]

    # Operators in the problem Alice, Bob and Eve
    A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
    B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
    Z = ncp.generate_operators('Z', 2, hermitian=0)

    substitutions = {}            # substitutions to be made (e.g. projections)
    moment_ineqs = []            # Moment inequalities (e.g. Tr[rho CHSH] >= c)
    moment_eqs = []                # Moment equalities (not needed here)
    op_eqs = []                    # Operator equalities (not needed here)
    op_ineqs = []                # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
    extra_monos = []            # Extra monomials to add to the relaxation beyond the level.


    # Get the relevant substitutions
    substitutions = get_subs()

    # Define the moment inequality related to chsh score
    score_cons = score_constraints(test_score_PV[d-3],d,YO=False)

    # Get any extra monomials we wanted to add to the problem
    extra_monos = get_extra_monomials()

    # Define the objective function (changed later)
    obj = objective(1)

    # Finally defining the sdp relaxation in ncpol2sdpa
    ops = ncp.flatten([A,B,Z])
    sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
    sdp.get_relaxation(level = LEVEL,
        equalities = op_eqs[:],
        inequalities = op_ineqs[:],
        momentequalities = moment_eqs[:],
        momentinequalities = moment_ineqs[:] + score_cons[:],
        objective = obj,
        substitutions = substitutions,
        extramonomials = extra_monos)

    # # Test
    ent = compute_entropy(sdp)
    print(f"SDP bound for d = {d}: {ent}")









