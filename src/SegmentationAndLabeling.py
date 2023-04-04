import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional

from docplex.mp.model import Model

try:
    from openqaoa.problems.converters import FromDocplex2IsingModel
except:
    print("Warning! openQAOA is not installed!")


class SegmentationAndLabeling:
    """Implementation of the Segmentation&Labeling problem."""

    def __init__(
        self,
        num_nodes: int,
        num_segments: int,
        num_labels: int,
        A: np.ndarray,
        B: np.ndarray,
        pos: Optional[dict] = None,
    ) -> None:
        self.num_nodes = num_nodes

        self.num_segments = num_segments
        if self.num_segments > self.num_nodes:
            raise ValueError(
                "Number of segments (clusters) cannot exceed number of nodes!"
            )

        self.num_labels = num_labels

        self.A = A
        self.B = B

        # for the graph setup and visualization
        self.pos = pos

        # setup the graph from the model
        self._build_graph()

    def _build_graph(self) -> None:
        """Convert input data into graph parameters"""
        self.G = nx.Graph()
        # add nodes
        for d in range(self.num_nodes):
            s = f"{d}\n"
            for c in range(self.num_labels):
                s += f"{c}:{self.A[d, c]:.1f}\n"
            self.G.add_node(d, size=100, shape="box", label=s)
        # add edges
        for d in range(self.num_nodes):
            for d1 in range(d + 1, self.num_nodes):
                w = ""
                for c in range(self.num_labels):
                    for c1 in range(self.num_labels):
                        w += f" {self.B[d, d1, c, c1]:.1f}"
                self.G.add_edge(d, d1, weight=w)
        if self.pos is None:
            self.pos = nx.circular_layout(self.G)

    def draw_graph(
        self, size: tuple[int, int] = (5, 5), draw_edge_labels=False, label_pos=0.5
    ):
        """Variant of visualizing the S&L data with graph"""
        # network visualization options
        options = {
            "font_size": 12,
            "node_size": 2000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 1,
            "node_shape": "s",
            "alpha": 1.0,
        }
        # draw network
        fig = plt.figure(figsize=size, frameon=False)
        ax = fig.add_axes(rect=(0, 0, 1, 1), frameon=False)

        if self.num_labels == 1:
            options["width"] = 0  # disable drawing edges in draw_networkx
            edge_weights = [
                i for i in nx.get_edge_attributes(self.G, "weight").values()
            ]
            edge_colors = [
                "blue" if float(edge_weights[i]) < 0.0 else "red"
                for i, e in enumerate(self.G.edges)
            ]

            for i, (d, d1) in enumerate(self.G.edges):
                nx.draw_networkx_edges(
                    self.G,
                    self.pos,
                    edgelist=[(d, d1)],
                    width=2 * float(edge_weights[i]),
                    edge_color=edge_colors[i],
                    style="solid",
                    alpha=1.0,
                    edge_cmap=None,
                    ax=ax,
                    label=edge_weights[i],
                )

        node_labels = nx.get_node_attributes(self.G, "label")
        nx.draw_networkx(self.G, self.pos, labels=node_labels, ax=ax, **options)
        if draw_edge_labels:
            edge_labels = nx.get_edge_attributes(self.G, "weight")
            nx.draw_networkx_edge_labels(
                self.G, pos=self.pos, edge_labels=edge_labels, label_pos=label_pos
            )

        plt.show()

    def draw_solution(self, x: np.ndarray, size: tuple[int, int] = (5, 5)) -> None:
        """Visualize the solution on a graph: segments are encoded with colors and labels are encoded with shapes"""

        # cmap = plt.cm.get_cmap('prism', 20)
        # cluster_grid = [mpl.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        cluster_grid = ["r", "b", "g", "k", "c", "m", "y", "w"]
        label_grid = ["o", "s", "^", ">", "v", "<", "d", "p", "h", "8"]

        clusters = []
        labels = []

        i = 0
        for d in range(self.num_nodes):
            for c in range(self.num_labels):
                for s in range(self.num_segments):
                    if x[i] > 0.0:
                        labels.append(label_grid[c])
                        clusters.append(cluster_grid[s])
                    i += 1

        if len(clusters) != self.num_nodes:
            print("Warning! Invalid solution!")

        # Draw Graph
        fig = plt.figure(figsize=size, frameon=False)
        ax = fig.add_axes(rect=(0, 0, 1, 1), frameon=False)

        # Draw the nodes for each shape with the shape specified
        nx.draw_networkx_labels(self.G, self.pos, ax=ax)  # draw node indexes
        for i, node in enumerate(self.G.nodes()):
            nx.draw_networkx_nodes(
                self.G,
                self.pos,
                nodelist=[i],
                node_size=2000,
                node_color=clusters[i],
                node_shape=labels[i],
                edgecolors="black",
                linewidths=2,
                label=self.G.nodes.data(),
                alpha=0.8,
            )

        # Draw edges
        nx.draw_networkx_edges(self.G, self.pos, width=0.5)
        for d, d1 in self.G.edges:
            if clusters[d] == clusters[d1]:
                nx.draw_networkx_edges(
                    self.G,
                    self.pos,
                    edgelist=[(d, d1)],
                    width=4.0,
                    edge_color=clusters[d],
                    style="solid",
                    alpha=None,
                    edge_cmap=None,
                    ax=ax,
                    label=None,
                )
        # edge_labels = nx.get_edge_attributes(self.G, "weight")
        # nx.draw_networkx_edge_labels(self.G, pos=self.pos, edge_labels=edge_labels)

        plt.show()

    def build_model(
        self, A: float = 1.0, B: float = 1.0, C: float = 0.0, D: float = None
    ):
        """Build DOcplex binary problem formulation"""

        d_edges = [
            (d, d1)
            for d in range(self.num_nodes)
            for d1 in range(d + 1, self.num_nodes)
        ]
        c_pairs = [
            (c, c1) for c in range(self.num_labels) for c1 in range(self.num_labels)
        ]

        # Setup DOcplex model
        mdl = Model(name="Segmentation and Labeling")

        x = {
            (d, c, s): mdl.binary_var(name=f"x_{d}_{c}_{s}")
            for d in range(self.num_nodes)
            for c in range(self.num_labels)
            for s in range(self.num_segments)
        }

        term1 = mdl.sum(
            self.A[d, c] * x[d, c, s]
            for d in range(self.num_nodes)
            for c in range(self.num_labels)
            for s in range(self.num_segments)
        )

        term2 = mdl.sum(
            self.B[d, d1, c, c1] * x[d, c, s] * x[d1, c1, s]
            for d, d1 in d_edges
            for c, c1 in c_pairs
            for s in range(self.num_segments)
        )

        # extra penalization term for the same labels in the segment
        term3 = mdl.sum(
            np.abs(self.B[d, d1, c, c]) * x[d, c, s] * x[d1, c, s]
            for d, d1 in d_edges
            for c in range(self.num_labels)
            for s in range(self.num_segments)
        )

        term4 = mdl.sum(
            (
                1
                - mdl.sum(
                    x[d, c, s]
                    for c in range(self.num_labels)
                    for s in range(self.num_segments)
                )
            )
            ** 2
            for d in range(self.num_nodes)
        )

        #########################

        if D is None:
            a_max = np.amax(np.abs(np.nan_to_num(self.A)))
            b_max = np.amax(np.abs(np.nan_to_num(self.B)))
            D = np.max([a_max, b_max])
            # print('automatic D=', D)

        mdl.minimize(A * term1 + B * term2 + C * term3 + D * term4)

        return mdl

    @staticmethod
    def evaluate_objective(mdl, x):
        """Given a DOCplex or (openQAOA) Ising model, compute the objective function for a specified input x"""

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
