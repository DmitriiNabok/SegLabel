import numpy as np

from mixers import x_mixer, x_mixer_initial_state
from mixers import xy_mixer, xy_mixer_initial_state
from common import get_counts, linear_ramp_params_customized

from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA

from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance


def qaoa_solver(
    model, reps=1, mixer='X', init_params=None, backend=None, backend_options=None, n_shots=1024, n_reads=1024, 
    optimize_params=False, seed=12345, verbose=False):
    """Setup and solve the model using the QAOA algorithm"""
        
    # Build the DOcplex model
    mdl = model.build_model()
    
    # and solve it with CPLEX to get the classical reference
    mdl.solve()
    if verbose:
        mdl.print_information()
        mdl.print_solution()
        print("")
    
    obj_ref = mdl.objective_value
    x_ref = np.zeros(mdl.number_of_binary_variables)
    for v in mdl.iter_binary_vars():
        x_ref[v.index] = v.solution_value
        
    if mixer == 'XY':
        # Rebuild the DOcplex model by setting the penalty strength to zero
        mdl = model.build_model(C=0.0)
    
    # Setup the QUBO problem
    mdl_qubo = QuadraticProgramToQubo().convert(from_docplex_mp(mdl))
    qubitOp, offset = mdl_qubo.to_ising()
    if True:
        # normalize the Hamiltonian
        w_max = np.max(np.abs(qubitOp.primitive.coeffs))
        qubitOp.primitive.coeffs /= w_max
        offset /= w_max

    if verbose:
        print("Offset:", offset)
        q2 = mdl_qubo.objective.quadratic.to_dict()
        num_qubits = qubitOp.num_qubits
        num_q2 = len(q2) - num_qubits
        print("Number of quadratic terms: ", num_q2)
        print("QUBO matrix sparsity: ", num_q2 / (num_qubits * (num_qubits - 1) * 0.5))

    if mixer == "X":
        initial_state = x_mixer_initial_state(qubitOp.num_qubits)
        mixer = x_mixer(qubitOp.num_qubits)
    elif mixer == "XY":
        initial_state = xy_mixer_initial_state(qubitOp.num_qubits, model.num_nodes, model.num_labels * model.num_segments)
        mixer = xy_mixer(qubitOp.num_qubits, model.num_nodes, model.num_labels * model.num_segments)
    else:
        raise ValueError("Unknown mixer! Available options are 'X' and 'XY'.")    
    
    # QAOA ansatz
    qc = QAOAAnsatz(qubitOp, reps=reps, initial_state=initial_state, mixer_operator=mixer)
    
    # Backend setup
    if backend is None:    
        backend = Aer.get_backend("qasm_simulator")
        
    # Circuit parameter initialization
    if init_params is None:
        init_params = linear_ramp_params_customized(p=reps, beta_scale=0.5, gamma_scale=2.0)
    if verbose:
        print("Initial parameters: ", init_params)
        
    if optimize_params:        
        quantum_instance = QuantumInstance(
            backend, backend_options=backend_options, seed_simulator=seed, seed_transpiler=seed, shots=n_shots
        )

        qaoa = VQE(
            ansatz=qc,
            optimizer=COBYLA(maxiter=1000, tol=1e-4),
            initial_point=init_params,
            quantum_instance=quantum_instance,
        )

        qaoa_results = qaoa.compute_minimum_eigenvalue(qubitOp)
        params = list(qaoa_results.optimal_parameters.values())
        if verbose:
            print("Optimized parameters: ", params)
        
    else:
        params = init_params

    # QAOA circuit sampler
    counts = get_counts(
        qc,
        params,
        backend=backend, backend_options=backend_options,
        n_shots=n_reads,
        seed=seed,
    )
    
    sols = []
    for s, p in counts.items():
        x = [int(i) for i in reversed(list(s))]
        obj = mdl_qubo.objective.evaluate(x)
        sols.append({"x": x, "prob": p / n_reads, "obj": obj, "feas": model.is_valid(x)})
    sols = sorted(sols, key=lambda k: k["obj"]) # sort by objective values
    
    # compute success probability
    sprob = np.sum([item['prob'] for item in sols if np.isclose(item['obj'], obj_ref)])
    
    output = {
        "n_vars": qubitOp.num_qubits,
        "reps": reps, "n_shots": n_shots, "n_reads": n_reads, "backend": backend.name, 
        "parameters": params, 
        "success_prob": sprob, 
        "solutions": sols, 
        "obj_ref": obj_ref, 
        "x_ref": x_ref
    }

    return output
