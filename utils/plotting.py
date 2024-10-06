import matplotlib.pyplot as plt
import matplotlib.tri as tri
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


def plot_strains(strains, nodes, elements, title="Strain Plot", cmap="viridis", show_colorbar=True, vmin=None, vmax=None):
    """
    Plot element-wise strains on the mesh.
    
    Parameters:
    - strains: Numpy array of strain values for each element (shape: (num_elements, 3))
    - nodes: Numpy array of node coordinates (shape: (num_nodes, 2))
    - elements: Numpy array of element connectivity (shape: (num_elements, 3) or (num_elements, 4) for quad)
    - title: Title of the plot (default: "Strain Plot")
    - cmap: Colormap for the plot (default: "viridis")
    - show_colorbar: Whether to show colorbar (default: True)
    - vmin: Minimum value for colormap scaling (default: None, auto-scale)
    - vmax: Maximum value for colormap scaling (default: None, auto-scale)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Triangulate the mesh (works for both triangles and quadrilaterals)
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    # Compute the average strain for each element
    avg_strain = np.mean(strains, axis=1)
    
    # Plot the strains
    tpc = ax.tripcolor(triangulation, avg_strain, shading='flat', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add optional colorbar
    if show_colorbar:
        plt.colorbar(tpc, ax=ax)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_aspect('equal')
    
    plt.show()


def plot_stresses(stresses, nodes, elements, title="Stress Plot", cmap="plasma", show_colorbar=True, vmin=None, vmax=None):
    """
    Plot element-wise stresses on the mesh.
    
    Parameters:
    - stresses: Numpy array of stress values for each element (shape: (num_elements, 3))
    - nodes: Numpy array of node coordinates (shape: (num_nodes, 2))
    - elements: Numpy array of element connectivity (shape: (num_elements, 3) or (num_elements, 4) for quad)
    - title: Title of the plot (default: "Stress Plot")
    - cmap: Colormap for the plot (default: "plasma")
    - show_colorbar: Whether to show colorbar (default: True)
    - vmin: Minimum value for colormap scaling (default: None, auto-scale)
    - vmax: Maximum value for colormap scaling (default: None, auto-scale)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Triangulate the mesh (works for both triangles and quadrilaterals)
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    # Compute the average stress for each element
    avg_stress = np.mean(stresses, axis=1)
    
    # Plot the stresses
    tpc = ax.tripcolor(triangulation, avg_stress, shading='flat', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add optional colorbar
    if show_colorbar:
        plt.colorbar(tpc, ax=ax)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_aspect('equal')
    
    plt.show()

def plot_x_stress_isobars(stresses, nodes, elements, title="Isobar Plot", cmap="coolwarm", levels=None):
    """
    Plot isobar lines for stress distribution on the mesh.
    
    Parameters:
    - stresses: Numpy array of stress values for each element (shape: (num_elements, 3))
    - nodes: Numpy array of node coordinates (shape: (num_nodes, 2))
    - elements: Numpy array of element connectivity (shape: (num_elements, 3) or (num_elements, 4) for quad)
    - title: Title of the plot (default: "Isobar Plot")
    - cmap: Colormap for the plot (default: "coolwarm")
    - levels: Contour levels for isobars (default: None for automatic scaling)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Triangulate the mesh (works for both triangles and quadrilaterals)
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Compute the average stress for each node
    avg_stress = np.zeros(len(nodes))
    stress_count = np.zeros(len(nodes))

    for i, element in enumerate(elements):
        for node_index in element:
            avg_stress[node_index] += stresses[i, 0]  # Taking the first component (σ_x)
            stress_count[node_index] += 1

    # Avoid division by zero for nodes with no associated stress
    avg_stress[stress_count > 0] /= stress_count[stress_count > 0]

    # Create a contour plot for isobars
    contour = ax.tricontourf(triangulation, avg_stress, levels=levels, cmap=cmap)
    ax.tricontour(triangulation, avg_stress, levels=levels, colors='black', linewidths=0.5)  # Contour lines

    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Stress (σ_x)', fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_aspect('equal')

    plt.show()

def plot_y_stress_isobars(stresses, nodes, elements, title="Y-Stress Isobar Plot", cmap="coolwarm", levels=None):
    """
    Plot isobar lines for y-stress distribution on the mesh.
    
    Parameters:
    - stresses: Numpy array of stress values for each element (shape: (num_elements, 3))
    - nodes: Numpy array of node coordinates (shape: (num_nodes, 2))
    - elements: Numpy array of element connectivity (shape: (num_elements, 3) or (num_elements, 4) for quad)
    - title: Title of the plot (default: "Y-Stress Isobar Plot")
    - cmap: Colormap for the plot (default: "coolwarm")
    - levels: Contour levels for isobars (default: None for automatic scaling)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Triangulate the mesh (works for both triangles and quadrilaterals)
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Compute the average y-stress for each node
    avg_y_stress = np.zeros(len(nodes))
    stress_count = np.zeros(len(nodes))

    for i, element in enumerate(elements):
        for node_index in element:
            avg_y_stress[node_index] += stresses[i, 1]  # Taking the second component (σ_y)
            stress_count[node_index] += 1

    # Avoid division by zero for nodes with no associated stress
    avg_y_stress[stress_count > 0] /= stress_count[stress_count > 0]

    # Create a contour plot for y-stress isobars
    contour = ax.tricontourf(triangulation, avg_y_stress, levels=levels, cmap=cmap)
    ax.tricontour(triangulation, avg_y_stress, levels=levels, colors='black', linewidths=0.5)  # Contour lines

    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Y-Stress (σ_y)', fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_aspect('equal')

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
