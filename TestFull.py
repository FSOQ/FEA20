#Проверил всю структуру по заданию глобальной матрицы жесткости. Нужно разбираться с функцией решателя
import meshpy.triangle as triangle
import numpy as np
import pandas as pd

# Определение точек и фасетов
points = [(0, 10), (10, 10), (15, 5), (30, 5), (30, 0), (0, 0)]
facets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

# Функция для настройки критериев уточнения сетки
def refine(vertices, area):
    return area > 1

# Функция для создания сетки
def create_mesh():
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh = triangle.build(mesh_info, refinement_func=refine)
    return mesh

#
#
#

# Функция для задания граничных условий
def apply_boundary_conditions(mesh_points):
    x_coords = mesh_points[:, 0]
    y_coords = mesh_points[:, 1]

    x_min = min(x_coords)
    x_max = max(x_coords)


    # Using int() to ensure we get integer values
    fixed_x_nodes = [
        int(i) for i in range(len(x_coords))
        if (x_coords[i] == x_min or x_coords[i] == x_max) and y_coords[i] != 0
    ]

    fixed_xy_nodes = [int(i) for i in range(len(y_coords)) if y_coords[i] == 0]

    free_dof_indices = []
    for i in range(len(x_coords)):
        if i in fixed_xy_nodes:
            continue
        elif i in fixed_x_nodes:
            free_dof_indices.append(2 * i + 1)  # No need to convert here, 2*i + 1 will be integer
        else:
            free_dof_indices.append(2 * i)        # No need to convert here, 2*i will be integer
            free_dof_indices.append(2 * i + 1)    # No need to convert here, 2*i + 1 will be integer

    # Проверка, что все элементы в списках - это целые числа
    assert all(isinstance(x, int) for x in fixed_x_nodes), f"fixed_x_nodes contains non-integer values: {fixed_x_nodes}"
    assert all(isinstance(x, int) for x in fixed_xy_nodes), f"fixed_xy_nodes contains non-integer values: {fixed_xy_nodes}"
    assert all(isinstance(x, int) for x in free_dof_indices), f"free_dof_indices contains non-integer values: {free_dof_indices}"

    # Дополнительная проверка: списки не должны быть пустыми
    assert len(fixed_x_nodes) > 0, "fixed_x_nodes is empty"
    assert len(fixed_xy_nodes) > 0, "fixed_xy_nodes is empty"
    assert len(free_dof_indices) > 0, "free_dof_indices is empty"

    return fixed_x_nodes, fixed_xy_nodes, free_dof_indices


#
#
#

# Function to calculate area of a triangle
def triangle_area(p1, p2, p3):
    # Calculate the area of a triangle given its vertices
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def stress_from_strain(B, D, u):
    """
    Вычисление напряжений из деформаций с использованием матрицы B и D.
    
    B: матрица деформаций
    D: матрица упругости
    u: вектор перемещений
    
    Возвращает:
    sigma: вектор напряжений
    """
    strain = B @ u  # Деформации
    sigma = D @ strain  # Напряжения
    return sigma

def mohr_coulomb_stiffness_matrix(p1, p2, p3, c, phi, E, nu, plane_stress=True):
    """
    Вычисление локальной матрицы жесткости для модели Моора-Кулона.
    
    Параметры:
    p1, p2, p3: координаты вершин треугольника (каждая как (x, y)).
    c: когезия (прочность материала на сдвиг).
    phi: угол внутреннего трения материала (в градусах).
    E: модуль Юнга.
    nu: коэффициент Пуассона.
    u: вектор перемещений на элементе (6x1).
    plane_stress: bool, если True, рассчитывается для плоского напряжения, иначе плоской деформации.
    
    Возвращает:
    K_e: локальная матрица жесткости элемента.
    sigma: напряжения в элементе.
    """
    
    # Вычисление площади треугольника
    A = triangle_area(p1, p2, p3)

    # Проверка на корректность площади
    if A <= 0:
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

    # Матрица B для элемента
    B = np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3]
    ]) / (2 * A)


    # Модель упругости с учетом материала по Моору-Кулону
    if plane_stress:
        # Матрица упругости D для плоского напряжения
        D = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
    else:
        # Матрица упругости D для плоской деформации
        D = (E / ((1 + nu) * (1 - 2 * nu))) * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2]
        ])

    # Добавление параметров Моора-Кулона
    phi_rad = np.radians(phi)  # Угол трения в радианах
    tan_phi = np.tan(phi_rad)
    cohesion_term = c / tan_phi  # С учетом когезии и угла трения

    # Матрица жесткости K_e
    K_e = A * (B.T @ D @ B)

    # Добавление параметров модели Моора-Кулона (упрощенно)
    # Например, добавляем жесткость за счет когезии
    K_e += cohesion_term * np.eye(6)

    return K_e



#
#
#

# Функция для применения внешней нагрузки

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



# Сборка глобальной матрицы жесткости
def assemble_global_stiffness_matrix(mesh_points, mesh_elements, E, nu, c, phi):
    num_nodes = len(mesh_points)
    K = np.zeros((2 * num_nodes, 2 * num_nodes))
    
    for element in mesh_elements:
        nodes = [mesh_points[i] for i in element]
        K_e = mohr_coulomb_stiffness_matrix(*nodes, c, phi, E, nu)
        K_e_list.append(K_e)
        
        global_dof_indices = []
        for node in element:
            global_dof_indices.append(2 * node)
            global_dof_indices.append(2 * node + 1)
        
#        for i in range(6):
#            for j in range(6):
#                K[global_dof_indices[i], global_dof_indices[j]] += K_e[i // 2, j // 2]

                
#        for i in range(6):
#            for j in range(6):
#                global_i = global_dof_indices[i]
#                global_j = global_dof_indices[j]
#                local_i = i // 2
#                local_j = j // 2
                
#                K[global_i, global_j] += K_e[i, j]


        for i in range(6):
            for j in range(6):
                print(f"Global DOF indices: {global_dof_indices[i]}, {global_dof_indices[j]} - Local DOF: {i}, {j}")
                K[global_dof_indices[i], global_dof_indices[j]] += K_e[i, j]

    
    return K


#
#
#


import numpy as np

def solve_system(K, F, fixed_x_nodes, fixed_xy_nodes):
    num_dof = K.shape[0]
    
    # Граничные условия: объединение узлов с фиксированными перемещениями
    fixed_dof_indices = fixed_x_nodes + fixed_xy_nodes
    
    # Степени свободы для свободных узлов
    free_dof_indices = [i for i in range(num_dof) if i not in fixed_dof_indices]
    
    # Уменьшенная система для свободных степеней свободы
    K_reduced = K[np.ix_(free_dof_indices, free_dof_indices)]
    F_reduced = F[free_dof_indices]

    print(f'K-reduced shape: {K_reduced.shape}')
    print(f'F-reduced shape: {F_reduced.shape}')
    
    # Решение для свободных узлов
    U_free = np.linalg.solve(K_reduced, F_reduced)
    
    # Полный вектор перемещений U, включая фиксированные
    U = np.zeros(num_dof)
    U[free_dof_indices] = U_free
    
    # Присваиваем фиксированным степеням свободы их известные значения (например, ноль)
    U[fixed_x_nodes] = 0  # Если узлы зафиксированы по x
    U[fixed_xy_nodes] = 0  # Если узлы зафиксированы по x и y
    
    return U



#
#
#


import matplotlib.pyplot as plt

# Функция для построения сетки и граничных условий
def plot_mesh(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes):
    plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_elements, color='blue')
    plt.scatter(mesh_points[fixed_x_nodes, 0], mesh_points[fixed_x_nodes, 1], color='green', label='Fixed x displacement')
    plt.scatter(mesh_points[fixed_xy_nodes, 0], mesh_points[fixed_xy_nodes, 1], color='orange', label='Fixed x and y displacement')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mesh with Boundary Conditions')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()

# Функция для построения результатов
def plot_displacement(mesh_points, mesh_elements, U):
    displacements = np.sqrt(U[::2] ** 2 + U[1::2] ** 2)
    plt.tricontourf(mesh_points[:, 0], mesh_points[:, 1], mesh_elements, displacements, cmap='viridis')
    plt.colorbar(label='Displacement (m)')
    plt.scatter(mesh_points[:, 0], mesh_points[:, 1], color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Finite Element Displacement')
    plt.gca().set_aspect('equal')
    plt.show()


#
#
#


import pandas as pd

# Функция для вывода матрицы узлов
def print_node_matrix(mesh_points):
    df_nodes = pd.DataFrame(mesh_points, columns=['X', 'Y'])
    df_nodes.index.name = 'Node'
    print("\nNode Matrix (Coordinates of nodes):")
    print(df_nodes.to_string(index=True))

# Функция для вывода матрицы элементов
def print_element_matrix(mesh_elements):
    df_elements = pd.DataFrame(mesh_elements, columns=['Node1', 'Node2', 'Node3'])
    df_elements.index.name = 'Element'
    print("\nElement Matrix (Node indices for each element):")
    print(df_elements.to_string(index=True))

# Функция для вывода граничных условий
def print_boundary_conditions(fixed_x_nodes, fixed_xy_nodes):
    df_bc = pd.DataFrame({
        'Fixed X Nodes': pd.Series(fixed_x_nodes),
        'Fixed XY Nodes': pd.Series(fixed_xy_nodes)
    })
    print("\nBoundary Conditions:")
    print(df_bc.to_string(index=False))

# Функция для вывода локальных матриц жесткости
def print_local_stiffness_matrix(K_e_list):
    for i, K_e in enumerate(K_e_list):
        df_local_K = pd.DataFrame(K_e)
        print(f"\nLocal Stiffness Matrix for Element {i}:")
        print(df_local_K.to_string(index=False, header=False))

# Функция для вывода глобальной матрицы жесткости
def print_global_stiffness_matrix(K):
    df_global_K = pd.DataFrame(K)
    print("\nGlobal Stiffness Matrix:")
    print(df_global_K.to_string(index=False, header=False))

# Пример вызова всех функций после создания сетки и выполнения расчета
def display_fem_results(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes, K_e_list, K):
    # Вывод матрицы узлов
    print_node_matrix(mesh_points)
    
    # Вывод матрицы элементов
    print_element_matrix(mesh_elements)
    
    # Вывод граничных условий
    print_boundary_conditions(fixed_x_nodes, fixed_xy_nodes)
    
    # Вывод локальных матриц жесткости
    print_local_stiffness_matrix(K_e_list)
    
    # Вывод глобальной матрицы жесткости
    print_global_stiffness_matrix(K)

# Пример использования:
# После расчета FEM, например после сборки локальных и глобальных матриц
# K_e_list - это список всех локальных матриц жесткости
# K - это глобальная матрица жесткости

# Дополнительная функция для проверки локальных матриц жесткости
def check_local_stiffness_matrices(K_e_list):
    for i, K_e in enumerate(K_e_list):
        if np.linalg.cond(K_e) > 1e12:
            print(f"Warning: Local stiffness matrix for element {i} is nearly singular (condition number: {np.linalg.cond(K_e)})")
        if np.isclose(np.linalg.det(K_e), 0):
            print(f"Error: Local stiffness matrix for element {i} is singular (determinant is zero). Check element nodes or material properties.")

# Дополнительная функция для проверки граничных условий
def check_boundary_conditions(fixed_x_nodes, fixed_xy_nodes, num_nodes):
    if len(fixed_x_nodes) == 0 and len(fixed_xy_nodes) == 0:
        print("Warning: No boundary conditions applied! The system may be under-constrained.")
    if len(fixed_x_nodes) + len(fixed_xy_nodes) == num_nodes * 2:
        print("Error: All degrees of freedom are constrained. The system is over-constrained.")
    if len(fixed_x_nodes) + len(fixed_xy_nodes) == 0:
        print("Error: No constraints applied. The system is completely free to move, which leads to singularity.")

# Дополнительная функция для проверки глобальной матрицы жесткости
def check_global_stiffness_matrix(K):
    # Проверка условного числа матрицы (чем больше, тем хуже обусловленность)
    cond_number = np.linalg.cond(K)
    print(f"Condition number of global stiffness matrix: {cond_number}")
    if cond_number > 1e12:
        print("Warning: Global stiffness matrix is nearly singular (high condition number).")
    if np.isclose(np.linalg.det(K), 0):
        print("Error: Global stiffness matrix is singular (determinant is zero).")




#
#
#



if __name__ == "__main__":
    # Параметры материала для Mohr-Coulomb
    E = 20e3  # Модуль Юнга (Па)
    nu = 0.3   # Коэффициент Пуассона
    c = 5e3    # Сцепление (Па)
    phi = 30   # Угол внутреннего трения (градусы)

    # Создание сетки
    mesh = create_mesh()
    mesh_points = np.array(mesh.points)
    mesh_elements = np.array(mesh.elements)

    # Применение граничных условий
    fixed_x_nodes, fixed_xy_nodes, free_dof_indices = apply_boundary_conditions(mesh_points)

    check_boundary_conditions(fixed_x_nodes, fixed_xy_nodes, len(mesh_points))
    
    K_e_list = []

    # Сборка глобальной матрицы жесткости
    K = assemble_global_stiffness_matrix(mesh_points, mesh_elements, E, nu, c, phi)

    check_local_stiffness_matrices(K_e_list)
    check_global_stiffness_matrix(K)

    # Применение равномерной нагрузки
    load_magnitude = 10e3
    F = apply_uniform_load(mesh_points, load_magnitude, fixed_xy_nodes)

    #Вывод матриц
    display_fem_results(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes, K_e_list, K)
    plot_mesh(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes)

    # Решение системы
    U_free = solve_system(K, F, fixed_x_nodes, fixed_xy_nodes)


    # Построение сетки и результатов

    plot_displacement(mesh_points, mesh_elements, U_free)
#
#
#
#
#
#
#
def create_mesh(points, facets):
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh = triangle.build(mesh_info, refinement_func=refine)
    return mesh

mesh = create_mesh(points, facets)
mesh_points = np.array(mesh.points)
mesh_elements = np.array(mesh.elements)

def apply_boundary_conditions(mesh_points):
    x_coords = mesh_points[:, 0]
    y_coords = mesh_points[:, 1]

    x_min = min(x_coords)
    x_max = max(x_coords)


    # Using int() to ensure we get integer values
    fixed_x_nodes = [
        int(i) for i in range(len(x_coords))
        if (x_coords[i] == x_min or x_coords[i] == x_max) and y_coords[i] != 0
    ]

    fixed_xy_nodes = [int(i) for i in range(len(y_coords)) if y_coords[i] == 0]

    free_dof_indices = []
    for i in range(len(x_coords)):
        if i in fixed_xy_nodes:
            continue
        elif i in fixed_x_nodes:
            free_dof_indices.append(2 * i + 1)  # No need to convert here, 2*i + 1 will be integer
        else:
            free_dof_indices.append(2 * i)        # No need to convert here, 2*i will be integer
            free_dof_indices.append(2 * i + 1)    # No need to convert here, 2*i + 1 will be integer

    # Проверка, что все элементы в списках - это целые числа
    assert all(isinstance(x, int) for x in fixed_x_nodes), f"fixed_x_nodes contains non-integer values: {fixed_x_nodes}"
    assert all(isinstance(x, int) for x in fixed_xy_nodes), f"fixed_xy_nodes contains non-integer values: {fixed_xy_nodes}"
    assert all(isinstance(x, int) for x in free_dof_indices), f"free_dof_indices contains non-integer values: {free_dof_indices}"

    # Дополнительная проверка: списки не должны быть пустыми
    assert len(fixed_x_nodes) > 0, "fixed_x_nodes is empty"
    assert len(fixed_xy_nodes) > 0, "fixed_xy_nodes is empty"
    assert len(free_dof_indices) > 0, "free_dof_indices is empty"

    return fixed_x_nodes, fixed_xy_nodes, free_dof_indices

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
            global_dof_indices.append(2 * node) # x - component
            global_dof_indices.append(2 * node + 1) # y - component
        
        #for i in range(6):
        #   for j in range(6):
        #       K[global_dof_indices[i], global_dof_indices[j]] += K_e[i // 2, j // 2]
        for i in range(6):
            for j in range(6):
                #print(f"Global DOF indices: {global_dof_indices[i]}, {global_dof_indices[j]} - Local DOF: {i}, {j}")
                K[global_dof_indices[i], global_dof_indices[j]] += K_e[i, j]

    
    return K, B_matrices, D

def SW_load(mesh_points, mesh_elements, rho, fixed_xy_nodes, g=9.81):


    num_nodes = len(mesh_points)
    
    # Initialize the global load vector
    #Apply for every DOF(NN*2)
    global_load_vector = np.zeros((num_nodes*2, 2))  # (Fx, Fy) for each node

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

def solve_system(K, F, fixed_x_nodes, fixed_xy_nodes):
    #Getting rows number ([1] for coloumns)
    num_dof = K.shape[0]
    
    # Граничные условия: объединение узлов с фиксированными перемещениями
    fixed_dof_indices = fixed_x_nodes + fixed_xy_nodes
    
    # Степени свободы для свободных узлов
    free_dof_indices = [i for i in range(num_dof) if i not in fixed_dof_indices]
    
    # Уменьшенная система для свободных степеней свободы
    #The system after elumination

    K_reduced = K[np.ix_(free_dof_indices, free_dof_indices)]
    
    # Initialize reduced force vector (1D array, as force components are per DOF)
    
    F_reduced = F[free_dof_indices]  # Extract Fx and Fy for free DOFs directly
    #F_reduced = F  # Extract Fx and Fy for free DOFs directly

    """
    F_reduced = np.zeros((len(free_dof_indices), 2)) 
     # Initialize the reduced force vector
    for i, dof_index in enumerate(free_dof_indices):
        F_reduced[i, 0] = F[dof_index, 0]  # Fx component
        F_reduced[i, 1] = F[dof_index, 1]  # Fy component
    """

    print(f'K-reduced shape: {K_reduced.shape}')
    print(f'F-reduced shape: {F_reduced.shape}')
    
    # Решение для свободных узлов
    U_free = np.linalg.solve(K_reduced, F_reduced)
    
    # Полный вектор перемещений U, включая фиксированные
    U = np.zeros(num_dof)
    
    for i, dof_index in enumerate(free_dof_indices):
        U[dof_index] = U_free[i, 1]  # Only using the y-component, assuming you want to ignore x    

    """"
    U[free_dof_indices] = U_free

        # Assign the free displacements to their corresponding DOFs

    for i, dof_index in enumerate(free_dof_indices):
        if dof_index % 2 == 0:
            # x-component (even indices)
            U[dof_index] = U_free[i]  # Assign x-component displacement
        else:
            # y-component (odd indices)
            U[dof_index] = U_free[i]  # Assign y-component displacement
    """
    # Присваиваем фиксированным степеням свободы их известные значения (например, ноль)
    #U[fixed_x_nodes] = 0  # Если узлы зафиксированы по x
    #U[fixed_xy_nodes] = 0  # Если узлы зафиксированы по x и y
    
    for node in fixed_x_nodes:
        U[2 * node] = 0  # Zero displacement in x-direction for fixed x nodes
    for node in fixed_xy_nodes:
        U[2 * node] = 0      # Zero displacement in x-direction for fully fixed nodes
        U[2 * node + 1] = 0  # Zero displacement in y-direction for fully fixed nodes

    print(f'U shape: {U.shape}')
    

    return U

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
        for j in range(element_nodes):
            # Assuming each node has 2 DOFs (x and y)
            dof_x = 2 * element[j]       # x DOF - add element[]
            dof_y = 2 * element[j] + 1   # y DOF - add element[]
            element_dof_indices.append(dof_x)
            element_dof_indices.append(dof_y)

        # Print the current element and its DOF indices
        print(f"Element {i}: DOF indices {element_dof_indices}")


        U_element = U[element_dof_indices]

        # Strain calculation: ε = B * U
        strain = np.dot(B, U_element)
        strains.append(strain)

    return np.array(strains)

K, B_matrices, D = assemble_global_stiffness_matrix(mesh_points, mesh_elements, E, nu)

    #check_local_stiffness_matrices(K_e_list)
    #check_global_stiffness_matrix(K)

    # Применение SW
F = SW_load(mesh_points, mesh_elements, rho, fixed_xy_nodes, g=9.81)


    # Решение системы
U = solve_system(K, F, fixed_x_nodes, fixed_xy_nodes)

strains = compute_strains(U, B_matrices, mesh_elements)
