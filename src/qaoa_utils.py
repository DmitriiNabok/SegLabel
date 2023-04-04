import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from collections import Counter
from docplex.mp.model import Model

try:
    from openqaoa.problems.converters import FromDocplex2IsingModel
    from openqaoa.algorithms import QAOA
    from openqaoa.backends import create_device
except ModuleNotFoundError:
    print("Warning! openQAOA is not installed!")


def evaluate_objective(mdl, x):
    """Evaluate the value of the objective function for a DOCplex or openQAOA Ising model."""

    if isinstance(mdl, FromDocplex2IsingModel):

        n = len(mdl.idx_terms)
        A = np.zeros(n)
        B = np.zeros((n, n))
        for key, value in mdl.qubo_dict.items():
            if isinstance(key, tuple):
                if len(key) == 1:
                    A[key[0]] = value  # linear
                elif len(key) == 2:
                    B[key[0], key[1]] = value
        const = mdl.constant

    elif isinstance(mdl, Model):

        n = mdl.number_of_binary_variables
        A = np.zeros(n)
        B = np.zeros((n, n))
        for v in mdl.objective_expr.iter_terms():
            A[v[0].index] = v[1]
        for v in mdl.objective_expr.iter_quad_triplets():
            B[v[0].index, v[1].index] = v[2]
        const = mdl.objective_expr.constant

    return const + A @ x + x @ B @ x


def normalization(problem, periodic=False):
    """Normalize coefficients of the Ising Hamiltonian."""
    max_weight = np.max(np.abs(problem.weights))
    new_problem = copy(problem)
    if periodic:
        new_problem.weights = [weight // max_weight for weight in new_problem.weights]
    else:
        new_problem.weights = [weight / max_weight for weight in new_problem.weights]
    new_problem.constant /= max_weight
    return new_problem


def solve_openqaoa(
    mdl,
    p,
    n_runs=10,
    n_shots=1024,
    method="cobyla",
    maxiter=100,
    tol=1e-4,
    norm=True,
    seed=1234,
):
    """Runs the QAOA multiple times with different random seeds."""

    qaoa = QAOA()

    # Ising model setup
    mdl_ising = FromDocplex2IsingModel(mdl)

    # device
    qiskit_device = create_device(location="local", name="qiskit.qasm_simulator")
    qaoa.set_device(qiskit_device)

    # classical optimizer properties
    qaoa.set_classical_optimizer(
        method=method,
        maxiter=maxiter,
        tol=tol,
        cost_progress=False,
        parameter_log=False,
    )

    # randomize the starting seeds
    np.random.seed(seed)
    seeds = np.random.randint(1, 2**16, size=n_runs)

    x = []
    obj = []
    prob = []
    angles = []

    for _i, _seed in enumerate(seeds):
        print(f"... running seed {_i+1} / {n_runs}", end="\r")

        # circuit properties
        qaoa.set_circuit_properties(
            seed=_seed,
            p=p,
            param_type="standard",
            mixer_hamiltonian="x",
            init_type="ramp",
        )

        # backend properties
        qaoa.set_backend_properties(
            n_shots=n_shots,
            seed_simulator=_seed,
            qiskit_simulation_method="automatic",
        )

        # compiling QAOA setup
        if norm:
            qaoa.compile(normalization(mdl_ising.ising_model))
        else:
            qaoa.compile(mdl_ising.ising_model)

        # optimization cycle
        qaoa.optimize()

        qaoa_sol = qaoa.result.lowest_cost_bitstrings(100)
        _x = [int(j) for j in qaoa_sol["solutions_bitstrings"][0]]
        x.append(_x)
        obj.append(evaluate_objective(mdl_ising, _x))
        prob.append(qaoa_sol["probabilities"][0])

        angles.append(qaoa.result.optimized["angles"])
    print("\rFinished iterating over seeds!\n")

    results = dict()
    results["x"] = x
    results["obj"] = obj
    results["prob"] = prob
    results["angles"] = angles
    results["n_qubits"] = qaoa.result.cost_hamiltonian.n_qubits
    results["backend"] = qaoa.backend_properties.asdict()
    results["circuit"] = qaoa.circuit_properties.asdict()
    results["optimizer"] = {
        "method": qaoa.optimizer.method,
        "tol": qaoa.optimizer.tol,
        "initial_params": qaoa.optimizer.initial_params,
        "options": qaoa.optimizer.options,
    }

    if mdl.solve_status is None:
        mdl.solve()
    results["docplex_obj"] = mdl.objective_value

    docplex_x = np.zeros(mdl.number_of_binary_variables)
    for v in mdl.iter_binary_vars():
        docplex_x[v.index] = v.solution_value
    results["docplex_x"] = docplex_x

    return results


def stat_info(results):
    """Prints the results statistics info."""
    exact = results["docplex_obj"]
    print("DOCplex solution:", exact)

    o = np.array(results["obj"])
    o_best = o.min()
    print("\nObjective:")
    print(f"     mean +- std: {o.mean()} +- {o.std()}")
    print(f"          r_mean: {o.mean()/exact} +- {o.std()/exact}")
    print(f"            best: {o_best}")
    print(f"          r_best: {o_best/exact}")

    p = np.array(results["prob"])
    print("\nProbability:")
    print(f"     mean +- std: {p.mean()} +- {p.std()}")
    i = np.where(o == o_best)[0]
    p_best = p[i].max()
    print(f"            best: {p_best}")
    n_shots = results["backend"]["n_shots"]
    print(f"             CoP: {p_best*n_shots}")

    return


def get_lowest_objectives(res, n=None):
    """Sort results wrt. objectives"""

    res_obj = sorted(res, key=lambda k: k["obj"])

    obj_best = res_obj[0]["obj"]

    counter = 0
    for v in res_obj:
        if v["obj"] == obj_best:
            counter += 1
        else:
            break
    res_obj_best = res_obj[: counter + 1]

    print("\nMost likely best objectives:")
    print(max(res_obj_best, key=lambda k: k["prob"]))

    if n is None:
        return res_obj
    else:
        return res_obj[:n]


def get_most_likely(res, n=None):
    """Sort results wrt. probabilities"""

    res_prob = sorted(res, key=lambda k: k["prob"])

    p_best = res_prob[-1]["prob"]

    counter = 0
    for v in res_prob[::-1]:
        if v["prob"] == p_best:
            counter += 1
        else:
            break

    res_prob_best = sorted(res_prob[-counter:], key=lambda k: k["obj"])
    for r in res_prob_best:
        print(r)

    if n is None:
        return res_prob[::-1]
    else:
        return res_prob[-n:]


def sampling_openqaoa(
    mdl, p=1, n_shots=1024, n_runs=1, opt_angles=None, norm=True, seed=None
):

    # QAOA ansatz parameters
    if opt_angles is None:
        np.random.seed(seed)
        opt_angles = np.random.uniform(-np.pi, np.pi, 2 * p)

    qaoa = QAOA()

    # Ising model setup
    mdl_ising = FromDocplex2IsingModel(mdl)

    # device
    qiskit_device = create_device(location="local", name="qiskit.qasm_simulator")
    qaoa.set_device(qiskit_device)

    solutions = Counter({})

    np.random.seed(seed)
    for _seed in np.random.randint(2**32, size=n_runs):

        # circuit properties
        qaoa.set_circuit_properties(
            seed=_seed,
            p=p,
            param_type="standard",
            mixer_hamiltonian="x",
            init_type="custom",
            variational_params_dict={
                "betas": opt_angles[:p],
                "gammas": opt_angles[p:],
            },
        )

        # backend properties (already set by default)
        qaoa.set_backend_properties(n_shots=n_shots, seed_simulator=_seed)

        # disable classical optimizer properties
        qaoa.set_classical_optimizer(maxiter=0)

        # compiling QAOA setup
        if norm:
            qaoa.compile(normalization(mdl_ising.ising_model))
        else:
            qaoa.compile(mdl_ising.ising_model)

        # fake optimization run
        qaoa.optimize()

        # collect samplers
        s = Counter(qaoa.result.optimized["measurement_outcomes"])
        solutions += s

    solutions = dict(solutions)

    # entire sampling size
    n = qaoa.backend_properties.n_shots * n_runs

    res = []
    for s, p in solutions.items():
        x = [int(i) for i in s]
        obj = evaluate_objective(mdl_ising, x)
        res.append({"x": x, "obj": obj, "prob": p / n})

    res = sorted(res, key=lambda k: k["obj"])

    return res
