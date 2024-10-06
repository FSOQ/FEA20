from src.mesh_generation import create_mesh
import numpy as np

def test_mesh_creation():
    mesh = create_mesh()
    assert isinstance(mesh.points, np.ndarray)
