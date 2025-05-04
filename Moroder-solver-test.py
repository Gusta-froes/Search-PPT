import os
import copy
import cvxpy as cp
import ncpol2sdpa as ncp  

def build_template(scenario_tuple, LEVEL=2, parallel=0, print_bool=False):
    """
    Build the SDP relaxation structure (template) for a given scenario.
    This structure (Probability object, substitutions, moment matrix structure)
    is independent of the numerical data.
    """
    X, Y, A, B = scenario_tuple
    A_config = [A for _ in range(X)]
    B_config = [B for _ in range(Y)]
    P = ncp.Probability(A_config, B_config)
    substitutions = P.substitutions

    # Use a dummy objective (zero) just to build the structure.
    dummy_obj = 0
    sdp_template = ncp.MoroderHierarchy(
        [ncp.flatten(P.parties[0]), ncp.flatten(P.parties[1])],
        verbose=1 if print_bool else 0,
        normalized=False,
        ppt=1,
        parallel=parallel
    )
    sdp_template.get_relaxation(LEVEL, objective=dummy_obj, substitutions=substitutions)
    return sdp_template, P, substitutions

def solve_line(scenario_tuple, line, LEVEL=2, print_bool=False, parallel=0, template=None, P=None, substitutions=None):
    """
    For a given data line and scenario template, compute the numerical objective,
    update the relaxation, convert to a CVXPY problem, and solve.
    """
    X, Y, A, B = scenario_tuple

    # Parse the data line into numbers.
    numbers = list(map(float, line.strip().split()))
    n_joint_prob = (A - 1) * X * (B - 1) * Y
    n_alice_prob = (A - 1) * X
    n_bob_prob = (B - 1) * Y
    if len(numbers) != n_joint_prob + n_alice_prob + n_bob_prob:
        raise Exception(f"Data is not in the CG notation, it should be {n_joint_prob+n_bob_prob+n_alice_prob}, but is {len(numbers)}")

    joint_prob = numbers[0:n_joint_prob]
    alice_marginal = numbers[n_joint_prob: n_joint_prob + n_alice_prob]
    bob_marginal = numbers[n_joint_prob + n_alice_prob:]

    # Compute the objective using the Probability object.
    obj = 0
    count = 0
    for a in range(A - 1):
        for x in range(X):
            obj -= alice_marginal[count] * P([a], [x], "A")
            count += 1
    count = 0
    for b in range(B - 1):
        for y in range(Y):
            obj -= bob_marginal[count] * P([b], [y], "B")
            count += 1
    count = 0
    for a in range(A - 1):
        for b in range(B - 1):
            for x in range(X):
                for y in range(Y):
                    obj -= joint_prob[count] * P([a, b], [x, y])
                    count += 1

    # Create a deep copy of the template (so that the heavy symbolic structure is reused)
    sdp_instance = copy.deepcopy(template)
    # Update the objective with the newly computed value.
    sdp_instance.set_objective(obj)
    # Convert the updated relaxation to a CVXPY problem.
    cvxpy_problem = sdp_instance.convert_to_cvxpy()
    # Solve the CVXPY problem using SCS with custom parameters. We have to do this because ncpol does not parse custom parameters to cvxpy solvers (despite of what the documentation says)
    solution = cvxpy_problem.solve(
        solver=cp.SCS,
        eps=1e-10,
        alpha=1.8,
        normalize=False,
        max_iters = 1000000,
        verbose=print_bool
    )
    return -solution

def process_scenario(scenario, LEVEL=2, parallel=0):
    """
    For the given scenario, build the template once and then iterate over each
    line in the data file. For each line, update the objective, solve the SDP, print
    its solution, and write it to an output file.
    """
    # Build file paths.
    base_path = r"\Users\gusta\OneDrive\Área de Trabalho\IC - Peter Brown\bell-ineqs"
    filename_base = f"{scenario[0]}{scenario[1]}{scenario[2]}{scenario[3]}"
    completed_path = os.path.join(base_path, f"{filename_base}_NLSwitch.txt")
    out_path = os.path.join(base_path, f"{filename_base}_Moroder2.txt")

    with open(completed_path, 'r') as file:
        lines = file.readlines()

    # Build the SDP template once.
    sdp_template, P, substitutions = build_template(scenario, LEVEL=LEVEL, parallel=parallel, print_bool=False)

    results = []
    for i, line in enumerate(lines):
        sol = solve_line(scenario, line, LEVEL=LEVEL, print_bool= True, parallel=parallel,
                         template=sdp_template, P=P, substitutions=substitutions)
        print(f"Line {i+1} solved; solution: {sol}")
        results.append(sol)

    # Save results to output file.
    with open(out_path, 'w') as file:
        for sol in results:
            file.write(str(sol) + '\n')
    return results

# Example usage:
if __name__ == '__main__':
    
    # Process scenario and print each solved line's result.
    scenarios = [(4,4,2,2),(3,2,3,3),(3,3,3,2),(6,3,2,2),(2,2,4,4),(3,3,4,2)]
    for scenario in scenarios:
        print(f"Processing scenario: {scenario}")
        process_scenario(scenario, LEVEL=2, parallel=0)
