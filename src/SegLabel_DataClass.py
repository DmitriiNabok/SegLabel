# joint_idx     0 - r ankle,
#               1 - r knee,
#               2 - r hip,
#               3 - l hip,
#               4 - l knee,
#               5 - l ankle,
#               6 - r wrist,
#               7 - r elbow,
#               8 - r shoulder,
#               9 - l shoulder,
#               10 - l elbow,
#               11 - l wrist,
#               12 - thorax,
#               13 - head top)


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
