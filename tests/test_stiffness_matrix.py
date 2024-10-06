from src.Data_process import mohr_coulomb_stiffness_matrix
import numpy as np

def test_stiffness_matrix():
    p1, p2, p3 = [0, 0], [1, 0], [0, 1]
    K = mohr_coulomb_stiffness_matrix(p1, p2, p3, 5000, 30, 20000, 0.3)
    assert K.shape == (6, 6)
