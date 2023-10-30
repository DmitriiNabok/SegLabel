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
    
    
def xy_mixer_initial_state(n_nodes: int, n_colors: int, suppress: bool=True):
    """W state setup within a single node"""
    
    if suppress:
        initial_state = QuantumCircuit(n_nodes * (n_colors+1))
        q = []
        for i in range(n_nodes):
            q += [n_colors * i + j for j in range(n_colors)]
            q += [i + n_nodes * n_colors]
        for n in range(n_nodes):
            construct_W_state(initial_state, q[n * (n_colors+1) : (n + 1) * (n_colors+1)])
    else:
        initial_state = QuantumCircuit(n_nodes * n_colors)
        q = initial_state.qubits
        for n in range(n_nodes):
            construct_W_state(initial_state, q[n * n_colors : (n + 1) * n_colors])
        
    return initial_state

        
def xy_mixer(n_nodes: int, n_colors: int, suppress: bool=True):
    """XY mixing operator setup"""
    
    if suppress:
        _n_colors = n_colors + 1
        q = []
        for i in range(n_nodes):
            q += [n_colors * i + j for j in range(n_colors)]
            q += [i + n_nodes * n_colors]
    else:
        _n_colors = n_colors
        q = [i for i in range(n_nodes * n_colors)]
        
    mixer = QuantumCircuit(n_nodes * _n_colors)
    connectivity = list(itertools.combinations(range(_n_colors), 2))

    beta = Parameter("β")

    for n in range(n_nodes):
        _n = n * _n_colors
        for i, j in connectivity:
            mixer.cx(q[_n + i], q[_n + j])
            mixer.crx(-2*beta, q[_n + j], q[_n + i])
            mixer.cx(q[_n + i], q[_n + j])
        # mixer.barrier()
    
    return mixer


def x_mixer_initial_state(n_qubits: int):
    """Set the initial state for the X mixer"""
    initial_state = QuantumCircuit(n_qubits)
    
    for n in range(n_qubits):
        initial_state.h(n)
        
    return initial_state


def x_mixer(n_qubits: int):
    """X mixing operator setup"""
    mixer = QuantumCircuit(n_qubits)
    beta = Parameter("β")
    for n in range(n_qubits):
        mixer.rx(-2 * beta, n)
        # mixer.barrier()
    return mixer