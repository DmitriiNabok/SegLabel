import numpy as np
import sys, os
import matplotlib.pyplot as plt

from qiskit import Aer, execute, transpile
from qiskit.utils import algorithm_globals


def linear_ramp_params(p: int, slope: float = 0.7, beta_sign: float = 1.0) -> np.ndarray:
    """Linear ramp scheme for the QAOA parameters initialization"""
    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be positive integer")
    if slope <= 0:
        raise ValueError("slope must be non-zero positive")

    time = slope * p
    # create evenly spaced timelayers at the centers of p intervals
    dt = time / p
    # fill betas, gammas_singles and gammas_pairs
    betas = np.linspace(
        (dt / time) * (time * (1 - 0.5 / p)), (dt / time) * (time * 0.5 / p), p
    )
    gammas = betas[::-1]
    
    # params = np.vstack((beta_sign * betas, gammas)).ravel(order="F")
    params = np.concatenate((beta_sign * betas, gammas), axis=0)
    
    return params


def linear_ramp_params_customized(p, beta_scale=0.7, gamma_scale=0.7, order=-1):
    """Custom linear ramp scheme for the QAOA parameters initialization"""

    s = np.linspace(0.5 / p, (1 - 0.5 / p), p)

    betas = beta_scale * s[::order]
    gammas = gamma_scale * s

    return np.concatenate((betas, gammas))


def get_counts(
    ansatz, params, backend=Aer.get_backend("qasm_simulator"), backend_options=None, n_shots=1024, seed=12345
):
    """ """
    qc = ansatz.copy()
    qc.measure_all()

    algorithm_globals.random_seed = seed
    job = execute(
        qc.assign_parameters(parameters=params),
        backend, backend_options=backend_options,
        shots=n_shots,
        seed_simulator=seed,
        seed_transpiler=seed,
    )
    counts = job.result().get_counts()

    return counts


def plot_solutions(dict_sols, exact, width=0.1, show_feasible=False, sort_obj=True, x_range=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    for label, sol in dict_sols.items():
        if sort_obj:
            sol = sorted(sol, key=lambda d: d["obj"])

        x = [s["obj"] for s in sol]
        y = [s["prob"] for s in sol]
        z = [s["feas"] for s in sol]

        ax[0].plot(x, color="blue")
        ax[0].set_xlabel("Samples")
        ax[0].set_ylabel("Objective")

        ax[1].bar(x, y, width=width, label=label, color="blue")
        if show_feasible:
            x_feas, y_feas = [], []
            for _x, _y, _z in zip(x, y, z):
                if _z:
                    x_feas.append(_x)
                    y_feas.append(_y)
            ax[1].bar(
                x_feas, y_feas, width=width, label=label + " (feas)", color="green"
            )

        ax[1].set_xlabel("Objective", fontsize=16)
        ax[1].set_ylabel("Quasi probability", fontsize=16)

    # exact solution
    ax[0].axhline(y=exact, ls=":", lw=2, color="r")
    ax[1].axvline(x=exact, ls=":", lw=2, color="r", label="exact")
    
    if x_range is not None:
        ax[1].set_xlim(x_range)

    plt.legend()
    plt.show()


def get_resources_info(ansatz, backend, p=1, seed=12345):
    """ """

    n_runs = 10

    np.random.seed(seed)
    seeds = np.random.randint(0, 2**16, size=(n_runs,))

    resources = []
    for _seed in seeds:
        algorithm_globals.random_seed = _seed
        _ansatz = transpile(
            ansatz,
            backend,
            seed_transpiler=_seed,
        )
        resources.append(
            {
                "p": p,
                "num_qubits": _ansatz.num_qubits,
                "depth": _ansatz.depth(),
                "gates": list(_ansatz.count_ops().items()),
            }
        )
    return resources


def count_avg_resources(resources):
    """ """
    n_qubits = resources[0]["num_qubits"]
    depth = []

    gates = dict()
    for k in dict(resources[0]["gates"]).keys():
        gates[k] = []

    for entry in resources:
        depth.append(entry["depth"])
        d = dict(entry["gates"])
        for k, v in d.items():
            gates[k].append(v)

    print("\nResources:")
    print("p:", resources[0]["p"])
    print("n_qubits:", resources[0]["num_qubits"])
    print("depth:", np.mean(depth), np.std(depth))
    for k, v in gates.items():
        print(k, np.mean(v), np.std(v))