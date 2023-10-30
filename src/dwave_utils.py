import numpy as np
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import minorminer


def plot_qpu_pegasus_embedding(embedding, figsize=(12, 12), xzoom=0.02, yzoom=0.02):
    """Helper function to visualize the solution of a problem on a D-Wave processor with Pegasus topology"""

    physical_qubits = [k for chain in embedding.values() for k in chain]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    dnx.draw_pegasus(
        dnx.pegasus_graph(16),
        ax=ax,
        crosses=True,
        node_size=30,
        node_color="red",
        edge_color="gray",
    )
    dnx.draw_pegasus(
        dnx.pegasus_graph(16, node_list=physical_qubits),
        ax=ax,
        crosses=True,
        node_color="blue",
        node_size=70,
    )
    plt.margins(x=-xzoom, y=-yzoom)
    plt.show()


def plot_qubo_matrix(Q, figsize=(11, 6), vmin=-1, vmax=1):
    """QUBO matrix visualization"""
    dim = max(max(k, l) for k, l in Q) + 1
    Qmatrix = np.zeros((dim, dim))
    for (k, l), Qkl in Q.items():
        if k > l:  # aggregate terms in the upper triangle
            k, l = l, k
        Qmatrix[k, l] += Qkl
    plt.figure(figsize=figsize)
    plt.imshow(Qmatrix, cmap="bwr", vmin=vmin, vmax=vmax)
    # plt.clim((-2,2))
    plt.colorbar()
    plt.show()


def max_chain_length(embedding: dict) -> int:
    max_ = 0
    for _, chain in embedding.items():
        if len(chain) > max_:
            max_ = len(chain)
    return max_


def get_embedding_with_short_chain(
    bqm, sampler, num_tries: int = 5, verbose=True, seed=None,
) -> dict:
    """Try a few probabilistic embeddings and return the one with the shortest
    chain length

    Parameters:
        bqm: BinaryQuadraticModel,
        sampler: DWaveSampler,
        num_tries: Number of probabilistic embeddings
        verbose: Whether to print out diagnostic information
        seed: Random seed

    Returns:
        embedding: The minor embedding with the shortest chain length
    """

    embedding = None
    best_chain_length = 1000
    
    np.random.seed(seed)
    seeds = np.random.randint(0, 2**16, size=num_tries)

    for i_try in range(num_tries):
        try:
            _, topology, _ = sampler.structure
            emb = minorminer.find_embedding(
                bqm.quadratic,
                topology,
                verbose=0,
                random_seed=seeds[i_try],
                timeout=int(1e4),
                threads=4,
            )
            chain_length = max_chain_length(emb)
            if verbose:
                print(i_try, "max_chain_length:", chain_length)
            if (chain_length > 0) and (chain_length < best_chain_length):
                embedding = emb
                best_chain_length = chain_length
        except:
            pass

    if embedding is None:
        raise Exception("Could not find embedding")

    print("\nINFO:")
    print("\tMax chain length in best embedding:", max_chain_length(embedding))
    print("\tNumber of physical qubits in best embedding:", 
        sum(len(chain) for chain in embedding.values()),
    )

    return embedding
