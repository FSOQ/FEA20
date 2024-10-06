import numpy as np

def apply_uniform_load(mesh_points, load_magnitude, fixed_xy_nodes):
    num_nodes = len(mesh_points)
    F = np.zeros(2 * num_nodes)
    
    # Находим максимальную координату Y
    max_y = np.max(mesh_points[:, 1])
    
    # Применение нагрузки в y-направлении только к верхним узлам
    for i in range(num_nodes):
        if i not in fixed_xy_nodes and mesh_points[i, 1] == max_y:
            F[2 * i + 1] = load_magnitude  # Применяем нагрузку только к верхним свободным узлам
    
    return F
