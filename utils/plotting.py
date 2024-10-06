import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
