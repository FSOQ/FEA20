import numpy as np

# Function to calculate area of a triangle
def triangle_area(p1, p2, p3):
    # Calculate the area of a triangle given its vertices
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def D_matrix(E, nu, plane_stress=True):
 # Модель упругости с учетом материала по Моору-Кулону
    if plane_stress:
        # Матрица упругости D для плоского напряжения
        #sigma(z) = 0
        D = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
    else:
        # Матрица упругости D для плоской деформации
        #eps(z) = 0
        D = (E / ((1 + nu) * (1 - 2 * nu))) * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2]
        ])
    return D

def local_stiffness_matrix(p1, p2, p3, E, nu, plane_stress=True):
    
    # Вычисление площади треугольника
    A = triangle_area(p1, p2, p3)

    # Проверка на корректность площади
    if A <= 0.1:
        raise ValueError(f"Треугольник с площадью {A} вырожден. Точки: {p1}, {p2}, {p3}")

    # Извлечение координат
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Вычисление коэффициентов для матрицы B
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    D = D_matrix(E, nu)

    # Матрица B для элемента
    B = np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3]
    ]) / (2 * A)

    # Матрица жесткости K_e
    K_e = A * (B.T @ D @ B)

    # Добавление параметров модели Моора-Кулона (упрощенно)
    # Например, добавляем жесткость за счет когезии

    return K_e, B, D

# Сборка глобальной матрицы жесткости
def assemble_global_stiffness_matrix(mesh_points, mesh_elements, E, nu):
    
    num_nodes = len(mesh_points)
    num_elements = len(mesh_elements)

    K = np.zeros((2 * num_nodes, 2 * num_nodes))
        # To store B and D matrices for each element
    B_matrices = np.zeros((num_elements, 3, 6))  # Assuming each B is 3x6 for 2D elements


    for idx, element in enumerate(mesh_elements):
        # Extract local node coordinates for this element
        nodes = [mesh_points[i] for i in element]
        K_e, B, D = local_stiffness_matrix(*nodes, E, nu)  # Local stiffness matrix
        
        B_matrices[idx] = B

        
        global_dof_indices = []

        for node in element:
             global_dof_indices.extend([2 * node, 2 * node + 1])
        
        #    global_dof_indices.append(2 * node) # x - component
        #    global_dof_indices.append(2 * node + 1) # y - component
        
        #for i in range(6):
        #   for j in range(6):
        #       K[global_dof_indices[i], global_dof_indices[j]] += K_e[i // 2, j // 2]
        for i in range(6):
            for j in range(6):
                #print(f"Global DOF indices: {global_dof_indices[i]}, {global_dof_indices[j]} - Local DOF: {i}, {j}")
                K[global_dof_indices[i], global_dof_indices[j]] += K_e[i, j]

    
    return K, B_matrices, D


def stress_from_strain(B_matrices, D_matrices, U):
    """
    Вычисление напряжений из деформаций с использованием матрицы B и D.
    
    B: матрица деформаций
    D: матрица упругости
    u: вектор перемещений
    
    Возвращает:
    sigma: вектор напряжений
    """
    strain = B_matrices @ U  # Деформации
    sigma = D_matrices @ strain  # Напряжения
    return sigma

def compute_strains(U, B_matrices, mesh_elements):
    """
    Compute the strain vector for each element.
    
    Parameters:
    - U: Displacement vector (shape: (num_dof,))
    - B_matrices: List of strain-displacement matrices [B] for each element (shape: (3, 2 * num_nodes_per_element))

    Returns:
    - strains: List of strain vectors for each element (shape: (num_elements, 3))
    """
    num_elements = len(B_matrices)
    strains = []

   
    
        # Loop through each element and compute strain


    for i in range(num_elements):
        B = B_matrices[i]

        element = mesh_elements[i]  # Access the element for current index
        element_nodes = len(element)  # Number of nodes in this element
        #element_nodes = B.shape[1] // 2  # Number of nodes per element
        element_dof_indices = []
        # Construct the global DOF indices for the current element
        for node in element:
            # Assuming each node has 2 DOFs (x and y)
            element_dof_indices.extend([2 * node, 2 * node + 1])
#        for j in range(element_nodes):       
        #    dof_x = 2 * element[j]       # x DOF - add element[]
        #    dof_y = 2 * element[j] + 1   # y DOF - add element[]
        #    element_dof_indices.append(dof_x)
        #    element_dof_indices.append(dof_y)

        # Print the current element and its DOF indices
        print(f"Element {i}: DOF indices {element_dof_indices}")


        U_element = U[element_dof_indices]

        # Strain calculation: ε = B * U
        strain = np.dot(B, U_element)
        strains.append(strain)

    return np.array(strains)


def compute_stresses(strains, D):
    """
    Compute the stress vector for each element based on the strain vector.
    
    Parameters:
    - strains: List of strain vectors for each element (shape: (num_elements, 3))
    - D: Constitutive matrix (shape: (3, 3))

    Returns:
    - stresses: List of stress vectors for each element (shape: (num_elements, 3))
    """
    num_elements = strains.shape[0]
    stresses = []

    # Loop through each element and compute stress
    for i in range(num_elements):
        strain = strains[i]

        # Stress calculation: σ = D * ε
        stress = np.dot(D, strain)
        stresses.append(stress)

    return np.array(stresses)