import numpy as np
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def F_gate(circ: QuantumCircuit, q: list, i: int, j: int, n: int, k: int):
    """F gate"""
    theta = np.arccos(np.sqrt(1 / (n - k + 1)))
    circ.ry(-theta, q[j])
    circ.cz(q[i], q[j])
    circ.ry(theta, q[j])
    # circ.barrier()


def construct_W_state(circ: QuantumCircuit, q: list):
    """W state for the list of input qubits"""
    circ.x(q[-1])

    n = len(q)
    for i in range(n - 1):
        F_gate(circ, q, n - i - 1, n - i - 2, n, i + 1)
        # circ.barrier()

    for i in range(n - 1):
        circ.cx(q[n - i - 2], q[n - i - 1])


def xy_mixer_initial_state(n_qubits: int, n_nodes: int, n_labels: int):
    """W state setup within a single problem node"""
    initial_state = QuantumCircuit(n_qubits)
    q = initial_state.qubits

    for n in range(n_nodes):
        construct_W_state(initial_state, q[n * n_labels : (n + 1) * n_labels])

    return initial_state


def xy_mixer(n_qubits: int, n_nodes: int, n_labels: int):
    """XY mixing operator setup"""
    mixer = QuantumCircuit(n_qubits)

    connectivity = list(itertools.combinations(range(n_labels), 2))
    # print(connectivity)

    beta = Parameter("β")

    for n in range(n_nodes):
        _n = n * n_labels
        for i, j in connectivity:
            mixer.cx(_n + i, _n + j)
            mixer.crx(beta, _n + j, _n + i)
            mixer.cx(_n + i, _n + j)
        # mixer.barrier()

    return mixer
