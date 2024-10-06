from src.boundary_conditions import apply_boundary_conditions
import numpy as np

def test_boundary_conditions():
    points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    fixed_x_nodes, fixed_xy_nodes, free_dof_indices = apply_boundary_conditions(points)
    assert len(fixed_x_nodes) > 0
    assert len(fixed_xy_nodes) > 0
