# QUBO formulation of the Segmentation and Labeling Problem

The repository provides collection of tools to setup and solve the Segmentation 
and Labeling Problem (SLP). 

Algorithms:
* With or without the node suppression. The node suppression adds a functionality to
  discard contributions from some nodes.
* `NEW_VERSION` suffix indicates an alternative encoding to inlude the node
  suppression.

Solvers:
* Classical: CPLEX, TABU, Simulated Annealing
* Quantum: QAOA with X and XY mixers, and the D-Wave's direct QPU
  solver.

The `results` folder includes examples of applying the tools for solving SLP
using the classical and quantum algorithms.

Use-cases:
* `hip`, `hip2`: the smallest use-cases with 16 and 24 variables,
  correspondingly. Solved with both the classical and quantum solvers for
  comparison.
* `hip3` is a larger use-case with 36 qubits that is solved with classical
  solvers to benchmark 2 encodings to include the detection node suppression
  mechanism.
* `full`: the implemented forkflow is applied to solve (classically) the real 
  person body-part image recognition task.


## Prerequisites

Check out and install dependencies:

```
pip install -r requirements.txt
```

For running CPLEX and QAOA (does not work with the latest Qiskit versions):

```
pip install qiskit qiskit-optimization[cplex] matplotlib pickle
```

For running TABU, Simulated Annealing, and D-Wave QPU:

```
pip install dwave-ocean-sdk
```

