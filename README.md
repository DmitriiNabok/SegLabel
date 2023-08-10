# Segmentation and Labelling Problem

The Segmentation and Labelling Problem (SLP) in the QUBO formulation derived using the one-hot encoding.

Examples demonstrate the basic execution workflow. 
"Toy" examples also serve as the implementation tests. 
Restricted CPLEX solver from the Qiskit suite is used as the classical QUBO solver.
  
## Requirements

```bash
pip install qiskit qiskit-optimization[cplex] # for building DOcplex model and runnning CPLEX (for <1000 variables)
pip install matplotlib pickle networkx
```