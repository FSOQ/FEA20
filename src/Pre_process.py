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


def SW_load(mesh_points, mesh_elements, rho, fixed_xy_nodes, g=9.81):


    num_nodes = len(mesh_points)
    num_dof = num_nodes * 2
    # Initialize the global load vector
    #Apply for every DOF(NN*2)
    global_load_vector = np.zeros(num_dof,)  # (Fx, Fy) for each node # --- ВАЖНО ---- Стал одномерным массив

    # Loop through each element
    for element in mesh_elements:

        nodes = [mesh_points[i] for i in element]
        
        # Calculate the area of the element (assume triangular element)
        area = triangle_area(*nodes)

        # Self-weight force per node for this element
        load_per_node = (rho * g * area) / len(element)

        # Apply the self-weight load to each node in the element
        for node_index in element:
            if node_index not in fixed_xy_nodes:
                # Apply the load only in the y-direction (gravity)
                global_load_vector[2* node_index, 1] += load_per_node #Add 2 *

    return global_load_vector

def triangle_area(p1, p2, p3):
    # Calculate the area of a triangle given its vertices
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


