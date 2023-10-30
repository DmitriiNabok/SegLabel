import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional
from docplex.mp.model import Model


class SLP_Model:
    """Implementation of the Segmentation&Labeling problem."""

    def __init__(
        self,
        num_nodes: int,
        num_labels: int,
        num_segments: int,
        A: np.ndarray,
        B: np.ndarray,
        pos: Optional[dict] = None,
        name: Optional[str] = None,
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
        
        # (optional) the model's name
        self.name = name


    def build_model(self, A: float = 1.0, B: float = 1.0, C: float = 0.0, D: float = None, constrained=False):
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
        if self.name is None:
            mdl = Model(name="Segmentation and Labeling")
        else:
            mdl = Model(name=self.name)

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
            x[d, c, s] * x[d1, c, s]
            for d, d1 in d_edges
            for c in range(self.num_labels)
            for s in range(self.num_segments)
        )
        
        term4 = mdl.sum(
            (1-mdl.sum(x[d, c, s] for c in range(self.num_labels) for s in range(self.num_segments)))**2
            for d in range(self.num_nodes)
        )

        #########################
        
        if constrained:
            mdl.minimize(A * term1 + B * term2 + C * term3)
            
            # build the model with explicit constraints
            for d in range(self.num_nodes):
                mdl.add_constraint(
                    mdl.sum(
                        x[d, c, s]
                        for c in range(self.num_labels)
                        for s in range(self.num_segments)
                    )
                    == 1
                )
        else:
            # unconstrained formulations
            if D is None:
                a_max = np.amax(np.abs(np.nan_to_num(self.A)))
                b_max = np.amax(np.abs(np.nan_to_num(self.B)))
                D = np.max([a_max, b_max])
                # print('automatic D=', D)
                
            mdl.minimize(A * term1 + B * term2 + C * term3 + D * term4)

        return mdl

    def is_valid(self, x):
        n = self.num_labels * self.num_segments
        for i in range(self.num_nodes):
            _x = x[i * n : (i + 1) * n]
            if sum(_x) != 1:
                return False
        return True
    

class SegLabel(object):
    def __init__(
        self,
        alphas,
        betas,
        map_DNNdet_reindex,
        map_DNNclass_reindex,
        person_dic,
        dets_prob_dic,
    ):
        self.num_detections = len(dets_prob_dic.keys())
        self.num_classes = alphas.shape[1]
        self.max_num_persons = len(person_dic.keys())
        self.alphas = alphas  # \alpha_d_c np.array((num_detections, num_classes))
        self.betas = betas  # \beta_d_d'_c_c' np.array((num_detections, num_detections, num_classes, num_classes))
        self.map_DNNdet_reindex = map_DNNdet_reindex
        self.map_DNNclass_reindex = map_DNNclass_reindex
        self.person_dic = person_dic
        """ person_dic = {
              0 : { # person 0
                  joint_type_1 : {det_idx1 : distance_to_joint_of_same_person, det_idx2 : distance_to_joint_of_same_person},
                  joint_type_2 : {det_idx1 : distance_to_joint_of_same_person, det_idx2 : distance_to_joint_of_same_person}
              },
              1 : { # person 1
                  joint_type_1 : {det_idx1 : distance_to_joint_of_same_person, det_idx2 : distance_to_joint_of_same_person},
                  joint_type_2 : {det_idx1 : distance_to_joint_of_same_person, det_idx2 : distance_to_joint_of_same_person}
                  }
          }
        """
        self.dets_prob_dic = dets_prob_dic
        """
          dets_prob_dic = {
            det1_idx1: { joint_type1: 0.9371519088745117, joint_type: 0.0 },
            32:        { 3: 0.8694389462471008, 13: 0.0 },
            50:        { 3: 0.0, 13: 0.7737213373184204 },
          }
        """
        
