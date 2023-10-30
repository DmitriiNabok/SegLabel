# Solving Segmentation and Labeling problem with QAOA

The repository provides the QUBO formulation of the Segmentation and Labeling
problem. It also includes the examples of solving selected (small) use-cases
with classical solvers (CPLEX, TABU, Simulated Annealing), QAOA (X and XY mixers), 
and the D-Wave's QPU.
As a demonstration, the implemented workflow is applied to solve (classically) 
the real person body-part image recognition task.


## Prerequisites

Check out and install the dependencies:

```
pip install -r requirements.txt
```

For running CPLEX and QAOA (does not work with the latest Qiskit versions):

```
pip install qiskit qiskit-optimization[cplex] matplotlib pickle
```

For running TABU and D-Wave QPU:

```
pip install dwave-ocean-sdk
```

