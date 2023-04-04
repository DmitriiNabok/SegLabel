import numpy as np
from typing import Callable, List, Tuple, Dict
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import Aer
import matplotlib.pyplot as plt


def linear_ramp_params(p: int, tau: float = 0.7, beta_sign: float = 1.0) -> np.ndarray:
    """Linear ramp scheme for the QAOA parameters initialization"""
    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer")
    if not isinstance(tau, float) or tau <= 0:
        raise ValueError("tau must be a positive float")

    time = tau * p
    # create evenly spaced timelayers at the centers of p intervals
    dt = time / p
    # fill betas, gammas_singles and gammas_pairs
    betas = np.linspace(
        (dt / time) * (time * (1 - 0.5 / p)), (dt / time) * (time * 0.5 / p), p
    )
    gammas = betas[::-1]
    params = np.vstack((beta_sign * betas, gammas)).ravel(order="F")
    return params


def solutions_sampler(
    obj_func: Callable[[np.ndarray], float],
    ansatz: QuantumCircuit,
    params: np.ndarray,
    backend=Aer.get_backend("qasm_simulator"),
    n_shots: int = 128,
    seed: int = 12345,
) -> Tuple[dict, List[dict]]:
    """Qiskit variational circuit sampler"""
    qc = ansatz.copy()
    qc.measure_all()

    job = execute(
        qc.assign_parameters(parameters=params),
        backend,
        shots=n_shots,
        seed_simulator=seed,
        seed_transpiler=seed,
    )
    counts = job.result().get_counts()

    sols = []
    for s, n in counts.items():
        x = np.asarray([int(y) for y in reversed(list(s))])
        obj = obj_func(x)
        sols.append({"x": x, "obj": obj, "prob": n / n_shots})

    return sols


def plot_solutions(
    dict_sols: Dict[str, List[Dict[str, float]]], exact: float, width: float = 0.1
) -> None:
    """Plot the solutions of a Qiskit variational circuit sampler."""
    if not isinstance(dict_sols, dict):
        raise TypeError("dict_sols must be a dictionary")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    for label, sol in dict_sols.items():
        x = [s["obj"] for s in sol]
        y = [s["prob"] for s in sol]

        ax[0].plot(x)
        ax[0].set_xlabel("Samples")
        ax[0].set_ylabel("Objective")

        ax[1].bar(x, y, width=width, label=label)
        ax[1].set_xlabel("Objective")
        ax[1].set_ylabel("Quasi probability")

    # exact solution
    ax[0].axhline(y=exact, ls=":", color="k")
    ax[1].axvline(x=exact, ls=":", color="k", label="exact")

    plt.legend()
    plt.show()
