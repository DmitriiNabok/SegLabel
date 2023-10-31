import numpy as np
import networkx as nx
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from typing import Optional
from docplex.mp.model import Model
    
from enum import IntEnum
from collections import namedtuple

class BodyParts(IntEnum):
    RA = 0
    RK = 1
    RH = 2
    LH = 3
    LK = 4
    LA = 5
    RW = 6
    RE = 7
    RS = 8
    LS = 9
    LE = 10
    LW = 11
    T = 12
    H = 13

COLORS = ["r", "b", "g", "k", "c", "m", "y", "w"]
SUPPRESSED = {"style": "*", "color": "lightblue", "name": "?"}
DEFAULT = {"style": "o", "color": "grey", "name": "f"}

EDGES = {
    (BodyParts.RA, BodyParts.RK): 1,
    (BodyParts.RK, BodyParts.RH): 1,
    (BodyParts.RH, BodyParts.LH): 1,
    (BodyParts.LH, BodyParts.LK): 1,
    (BodyParts.LK, BodyParts.LA): 1,
    (BodyParts.RW, BodyParts.RE): 1,
    (BodyParts.RE, BodyParts.RS): 1,
    (BodyParts.RS, BodyParts.LS): 1,
    (BodyParts.LS, BodyParts.LE): 1,
    (BodyParts.LE, BodyParts.LW): 1,
    (BodyParts.T, BodyParts.H): 1,
    (BodyParts.LH, BodyParts.T): 1,
}

Node = namedtuple("Node", "style name")
LABELS = [None] * len(BodyParts)
LABELS[BodyParts.RA] = Node("^", "RA")
LABELS[BodyParts.RK] = Node("d", "RK")
LABELS[BodyParts.RH] = Node("D", "RH")
LABELS[BodyParts.LH] = Node("D", "LH")
LABELS[BodyParts.LK] = Node("d", "LK")
LABELS[BodyParts.LA] = Node("^", "LA")
LABELS[BodyParts.RW] = Node("<", "RW")
LABELS[BodyParts.RE] = Node("p", "RE")
LABELS[BodyParts.RS] = Node(">", "RS")
LABELS[BodyParts.LS] = Node(">", "LS")
LABELS[BodyParts.LE] = Node("p", "LE")
LABELS[BodyParts.LW] = Node("<", "LW")
LABELS[BodyParts.T] = Node("s", "T")
LABELS[BodyParts.H] = Node("o", "H")


class SegmentationAndLabeling:
    """Implementation of the Segmentation&Labeling problem."""

    def __init__(
        self,
        A: np.ndarray, 
        B: np.ndarray,
        max_num_segments: int=1,
        suppress: Optional[bool] = True,
        pos: Optional[dict] = None,
        class_reindex: Optional[dict] = None,
    ) -> None:
        
        self.A = A
        self.B = B
        
        self.num_nodes, self.num_labels = self.A.shape
        self.num_segments = max_num_segments
        
        print("Segmentation and Labeling Problem initialization:")
        print("  Number of nodes: ", self.num_nodes)
        print("  Number of labels: ", self.num_labels)
        print("  Maximum number of segments: ", self.num_segments)
        
        # include the detection suppression
        self.suppress = suppress

        # for the graph setup and visualization
        self.pos = pos
        
        # body part reindexation map
        self.class_reindex = class_reindex
        if self.class_reindex is None:
            self.class_reindex = {i: i for i in range(self.num_labels)}

        # setup the graph from the model
        self._build_graph()


    @property
    def num_nodes(self):
        return self._num_nodes
    @num_nodes.setter
    def num_nodes(self, num_nodes):
        if isinstance(num_nodes, int):
            if num_nodes > 0:
                self._num_nodes = num_nodes
            else:
                raise ValueError("<num_nodes> has to be > 0")
        else:
            raise TypeError("<num_nodes> has to be type int")


    @property
    def num_labels(self):
        return self._num_labels
    @num_labels.setter
    def num_labels(self, num_labels):
        if isinstance(num_labels, int):
            if num_labels > 0:
                self._num_labels = num_labels
            else:
                raise ValueError("<num_labels> has to be > 0")
        else:
            raise TypeError("<num_labels> has to be type int")


    @property
    def num_segments(self):
        return self._num_segments
    @num_segments.setter
    def num_segments(self, num_segments):
        if isinstance(num_segments, int):
            if num_segments > 0:
                if num_segments > self.num_nodes:
                    raise ValueError("<num_segments> cannot exceed <num_nodes>!")
                else:
                    self._num_segments = num_segments
            else:
                raise ValueError("<num_segments> has to be > 0")
        else:
            raise TypeError("<num_segments> has to be type int")


    def _build_graph(self) -> None:
        """Convert input data into graph parameters"""
        self.G = nx.Graph()
        # add nodes
        for d in range(self.num_nodes):
            self.G.add_node(d, size=100, shape="box", label=None)
        if self.pos is None:
            self.pos = nx.circular_layout(self.G)
        # add edges
        for d in range(self.num_nodes):
            for d1 in range(d + 1, self.num_nodes):
                self.G.add_edge(d, d1, weight=None)

            
    def draw_graph(
        self, ax, draw_node_labels=False, draw_edge_labels=False, label_pos=0.5, options=None,
    ):
        """Variant of visualizing the S&L data with graph"""
        if options is None:
            # network visualization options
            options = {
                "font_size": 8,
                "node_size": 2000,
                "node_color": "white",
                "edgecolors": "black",
                "linewidths": 2,
                "width": 1,
                "node_shape": "s",
                "alpha": 1.0,
            }

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

        node_labels = None
        if draw_node_labels:
            # node_labels = nx.get_node_attributes(self.G, "label")
            node_labels = dict()
            for d in range(self.num_nodes):
                s = f"{d}\n"
                for c in range(self.num_labels):
                    s += f"{c}:{self.A[d, c]:.1f}\n"
                node_labels[d] = s
            
        nx.draw_networkx(self.G, self.pos, labels=node_labels, ax=ax, **options)
        if draw_edge_labels:
            # edge_labels = nx.get_edge_attributes(self.G, "weight")
            edge_labels = dict()
            for d in range(self.num_nodes):
                for d1 in range(d + 1, self.num_nodes):
                    w = ""
                    for c in range(self.num_labels):
                        for c1 in range(self.num_labels):
                            w += f" {self.B[d, d1, c, c1]:.1f}"
                    # self.G.add_edge(d, d1, weight=w)
                    edge_labels[(d, d1)] = w
            
            nx.draw_networkx_edge_labels(
                self.G, pos=self.pos, edge_labels=edge_labels, label_pos=label_pos, font_size=options['font_size'],
            )
    

    def suppressed_nodes(self, x: np.ndarray) -> list:
        """Returns the list of the suppression flags for each node. If suppression is not used, simply returns the list of zeros."""
        if self.suppress:
            # information about the suppression nodes is taken from the solution
            y = x[self.num_nodes * self.num_labels * self.num_segments :]
            assert len(y)==self.num_nodes, "Severe bug!!!"
        else:
            # no suppression (dummy array)
            y = [0] * self.num_nodes
        return y
        

    def is_valid(self, x):
        "Check if the solution string satisfies the constraints for the 'unsuppressed' nodes"
        y = self.suppressed_nodes(x)
        n = self.num_labels * self.num_segments
        for i in range(self.num_nodes):
            if y[i] == 0:
                _x = x[i * n : (i + 1) * n]
                if sum(_x) != 1:
                    return False
        return True

    
    def draw_solution(self, ax, x: np.ndarray, node_size=200, width=2, alpha=0.8) -> None:
        """Visualize the solution on a graph: segments are encoded with colors and labels are encoded with shapes"""

        # collect information about detection labels and segments
        segments = []
        labels = []
        names = []
        
        y = self.suppressed_nodes(x)
        
        n = self.num_labels * self.num_segments
        for d in range(self.num_nodes):
            # check if the node is suppressed
            if y[d] == 0:
                _x = x[d * n : (d + 1) * n]
                if sum(_x) == 1:
                    i = 0
                    for c in range(self.num_labels):
                        for s in range(self.num_segments):
                            if _x[i] > 0:
                                segments.append(COLORS[s])
                                labels.append(LABELS[self.class_reindex[c]].style)
                                names.append(LABELS[self.class_reindex[c]].name)
                            i += 1
                else: # does not satisfy the constraints
                    segments.append(DEFAULT["color"])
                    labels.append(DEFAULT["style"])
                    names.append(DEFAULT["name"])
            else: # suppressed
                segments.append(SUPPRESSED["color"])
                labels.append(SUPPRESSED["style"])
                names.append(SUPPRESSED["name"])
                    
        # Draw the nodes for each shape with the shape specified
        for i, node in enumerate(self.G.nodes()):
            nx.draw_networkx_nodes(
                self.G,
                self.pos,
                nodelist=[i],
                node_color=segments[i],
                node_shape=labels[i],
                label=None,
                node_size=node_size,
                alpha=alpha,
                linewidths=0.5,
            )
            
        # Draw edges
        nx.draw_networkx_edges(self.G, self.pos, width=0.0)
        for d, d1 in self.G.edges:
            if segments[d] == segments[d1] and segments[d] != DEFAULT["color"] and segments[d] != SUPPRESSED["color"]:
                nx.draw_networkx_edges(
                    self.G,
                    self.pos,
                    edgelist=[(d, d1)],
                    edge_color=segments[d],
                    edge_cmap=None,
                    ax=ax,
                    label=None,
                    width=width,
                    style="-",
                    alpha=alpha,
                    
                )

        # Relabel nodes
        nx.draw_networkx_labels(
            self.G, self.pos, 
            {i: n for i, n in enumerate(names)}, 
            font_size=14, 
            font_color="whitesmoke",
        )
        

    def build_model(self, C: float = None):
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
        
        if self.suppress:
            # add auxiliary variables to decide about the node contribution "suppression"
            y = {
                (d): mdl.binary_var(name=f"y_{d}")
                for d in range(self.num_nodes)
            }     

        term1 = mdl.sum(
            self.A[d, c] * x[d, c, s]
            for d in range(self.num_nodes)
            for c in range(self.num_labels)
            for s in range(self.num_segments)   # exclude suppressed detections
        )

        term2 = mdl.sum(
            self.B[d, d1, c, c1] * x[d, c, s] * x[d1, c1, s]
            for d, d1 in d_edges
            for c, c1 in c_pairs
            for s in range(self.num_segments)   # exclude suppressed detections
        )
        
        # one-hot encoding constraint
        if self.suppress:
            term3 = mdl.sum(
                (1.0 - mdl.sum(
                           x[d, c, s] 
                           for c in range(self.num_labels) 
                           for s in range(self.num_segments)
                       )
                     - y[d]
                )**2
                for d in range(self.num_nodes)
            )
        else:
            term3 = mdl.sum(
                (1.0 - mdl.sum(
                           x[d, c, s] 
                           for c in range(self.num_labels) 
                           for s in range(self.num_segments)
                       )
                )**2
                for d in range(self.num_nodes)
            )

        #########################
        
        if C is None:
            C = np.amax(np.abs(np.nan_to_num(self.B)))
            print('automatic C=max(B)=', C)

        mdl.minimize(term1 + term2 + C * term3)

        return mdl
    
    
    def build_qubo(
        self,
        C: float = None,     # constraint strength
        norm: bool = False,
    ) -> (dict, float):
        """Setup the SLP QUBO matrix"""
        
        if C is None:
            C = np.amax(np.abs(np.nan_to_num(self.B)))
            print('automatic C=max(B)=', C)
        
        # number of variables
        if self.suppress:
            num_vars = self.num_nodes * self.num_labels * self.num_segments + self.num_nodes
        else:
            num_vars = self.num_nodes * self.num_labels * self.num_segments

        # index mappings
        dcs2i = np.zeros((self.num_nodes, self.num_labels, self.num_segments), dtype=int)
        i = 0
        for d in np.arange(self.num_nodes):
            for c in np.arange(self.num_labels):
                for s in np.arange(self.num_segments):
                    dcs2i[d, c, s] = i
                    i += 1

        qubo = np.zeros((num_vars, num_vars))

        # alpha term
        for d in np.arange(self.num_nodes):
            for c in np.arange(self.num_labels):
                for s in np.arange(self.num_segments):
                    i = dcs2i[d, c, s]
                    qubo[i, i] = self.A[d, c] 

        # beta term
        d_edges = [(d, d1) for d in np.arange(self.num_nodes) for d1 in np.arange(d+1, self.num_nodes)]
        c_pairs = [(c, c1) for c in np.arange(self.num_labels) for c1 in np.arange(self.num_labels)]
        for (d, d1) in d_edges:
            for (c, c1) in c_pairs:
                for s in np.arange(self.num_segments):
                    qubo[dcs2i[d, c, s], dcs2i[d1, c1, s]] += self.B[d, d1, c, c1]

        ########################### one-hot constraints #############################
        
        offset = C * self.num_nodes    # constant term

        # -2Cx
        for i in np.arange(num_vars):
            qubo[i, i] -= 2.0*C
        
        s_pairs = [(s, s1) for s in np.arange(self.num_segments) for s1 in np.arange(self.num_segments)]
        for d in np.arange(self.num_nodes):     # Cx^2
            for (c, c1) in c_pairs:
                for (s, s1) in s_pairs:
                    qubo[dcs2i[d, c, s], dcs2i[d, c1, s1]] += C
                    
        if self.suppress:
            # additional terms related to the auxiliary variables
            for d in np.arange(self.num_nodes): # loop over nodes
                i1 = self.num_nodes * self.num_labels * self.num_segments + d # y_d index
                # 2 * x_dcs * y_d 
                for c in np.arange(self.num_labels):
                    for s in np.arange(self.num_segments):
                        i = dcs2i[d, c, s] # x_dcs index
                        qubo[i, i1] += 2*C
                # y_d ^2
                qubo[i1, i1] += C

        Q = dict()
        for i in np.arange(num_vars):
            Q[(i, i)] = qubo[i, i]
            for i1 in np.arange(i+1, num_vars):
                value = qubo[i, i1] + qubo[i1, i] # account for x_0*x_1 = x_1*x_0 symmetry
                if value:
                    Q[(i, i1)] = value
        
        if norm:
            w = np.unique(np.abs(list(Q.values())))
            w_max = np.max(w)
            for key in Q.keys():
                Q[key] /= w_max
            offset /= w_max

        return Q, offset
    
    
    def draw_skeleton(self, x: np.ndarray, node_size=200, width=2, alpha=0.8) -> None:
        """Draw the 'stick' model by combining the detection points by its label and segment"""

        n = self.num_labels * self.num_segments
        i2cs = np.zeros((n, 2), dtype=np.int32)
        i = 0
        for c in range(self.num_labels):
            for s in range(self.num_segments):
                i2cs[i, 0], i2cs[i, 1] = c, s
                i += 1

        nodes = dict()
        for d in range(self.num_nodes):
            _x = np.array(x[d * n : (d + 1) * n])
            if np.sum(_x) == 1:
                i = np.where(_x > 0)[0][0]
                if i in nodes:
                    nodes[i].append(d)
                else:
                    nodes[i] = [d]

        # compute averaged body part positions
        for d in nodes.keys():
            x, y = 0.0, 0.0
            for i in nodes[d]:
                x += self.pos[i][0]
                y += self.pos[i][1]
            x /= len(nodes[d])
            y /= len(nodes[d])
            nodes.update({d: [int(x), int(y)]})

        # visualize nodes and edges
        for d, p in nodes.items():
            c, s = i2cs[d, 0], i2cs[d, 1]
            # nodes
            plt.scatter(
                p[0],
                p[1],
                marker=LABELS[c].style,
                c=COLORS[s],
                s=200,
                alpha=0.6,
            )
            # edges
            for d1, p1 in nodes.items():
                c1, s1 = i2cs[d1, 0], i2cs[d1, 1]
                if (c, c1) in EDGES and s == s1:
                    ax = [p[0], p1[0]]
                    ay = [p[1], p1[1]]
                    plt.plot(ax, ay, "-", lw=4, c=COLORS[s], alpha=0.6)

