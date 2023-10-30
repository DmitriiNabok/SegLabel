import pickle
import numpy as np
import sys, os

sys.path.append(os.path.abspath("src"))
from SegLabel import SegLabel
from SegmentationAndLabeling import SegmentationAndLabeling
# from SegmentationAndLabeling_NEW_VERSION import SegmentationAndLabeling

def read_model(file, max_num_segments=None, suppress=False, verbose=False):
    """Read the SLP data and build the QUBO model"""
    
    # read the data
    with open(file, "rb") as f:
        segLabel = pickle.load(f)

    # beta_dc_d'c'u
    B = segLabel.betas.copy()
    B = np.nan_to_num(B, nan=0.0)
    B_max = np.max(B)
    if verbose:
        print("B:")
        print("  shape: ", np.shape(B))
        print("  min/max: ", np.min(B), B_max)

    # alpha_dc
    A = segLabel.alphas.copy()
    A = np.nan_to_num(A, nan=0.0)
    A_max = B_max
    A[A > A_max] = A_max  # truncate too high values
    if verbose:
        print("A:")
        print("  shape: ", np.shape(A))
        print("  min/max: ", np.min(A), np.max(A))
        print("")
        
    if max_num_segments is None:
        max_num_segments = segLabel.max_num_persons

    # Initialize the S&L problem
    model = SegmentationAndLabeling(
        A=A, B=B,
        max_num_segments=max_num_segments, 
        # suppress=suppress
    )
    
    return model