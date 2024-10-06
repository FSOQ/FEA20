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

    return K_e

# Сборка глобальной матрицы жесткости
def assemble_global_stiffness_matrix(mesh_points, mesh_elements, E, nu):
    
    num_nodes = len(mesh_points)
    K = np.zeros((2 * num_nodes, 2 * num_nodes))
    
    for element in mesh_elements:
        nodes = [mesh_points[i] for i in element]
        K_e = local_stiffness_matrix(*nodes, E, nu) # *Передача кортежем [array([]),array([]),array([])]
        #K_e_list.append(K_e)
        #Assembled local stiffness matrix
        
        global_dof_indices = []

        for node in element:
            global_dof_indices.append(2 * node)
            global_dof_indices.append(2 * node + 1)

        
        #        for i in range(6):
        #            for j in range(6):
        #                K[global_dof_indices[i], global_dof_indices[j]] += K_e[i // 2, j // 2]
        for i in range(6):
            for j in range(6):
                print(f"Global DOF indices: {global_dof_indices[i]}, {global_dof_indices[j]} - Local DOF: {i}, {j}")
                K[global_dof_indices[i], global_dof_indices[j]] += K_e[i, j]

    
    return K


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