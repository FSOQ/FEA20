from data.material_import import load_material_properties, get_material_properties
from src.mesh_generation import create_mesh, load_mesh_from_file
from src.boundary_conditions import apply_boundary_conditions
from src.Data_process import assemble_global_stiffness_matrix, compute_strains, compute_stresses
from src.Pre_process import SW_load
from src.solver import solve_system
from utils.plotting import plot_mesh, plot_displacement, display_fem_results, plot_stresses, plot_strains, plot_x_stress_isobars, plot_y_stress_isobars
#from tests.test_boundary_conditions import check_boundary_conditions
import numpy as np

if __name__ == "__main__":

    # Путь к файлу с точками
    points, facets = load_mesh_from_file('./data/mesh_file.csv')

    # Путь к файлу с материалами
    material_csv_path = "./data/material_properties.csv"
    # Загрузка свойств материала
    materials_df = load_material_properties(material_csv_path)

    # Пример выбора материала по имени
    material_name = "Soil_Mat_1"  # Заменить на нужное имя материала
    E, nu, c, phi, dilatancy, rho = get_material_properties(materials_df, material_name)

    print(f"Using Material: {material_name} with E={E}, nu={nu}, c={c}, phi={phi} dilatancy={dilatancy}, rho={rho}")

    # Создание сетки
    mesh = create_mesh(points, facets)
    mesh_points = np.array(mesh.points)
    mesh_elements = np.array(mesh.elements)

    #print(mesh_points)
    #print(mesh_elements)
    for element in mesh_elements:
        nodes = [mesh_points[i] for i in element]
        #print(f'Element {element} nodes: {nodes}')

    # Применение граничных условий
    fixed_x_nodes, fixed_xy_nodes, free_dof_indices = apply_boundary_conditions(mesh_points)
    #Вывод матриц
    #display_fem_results(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes, K_e_list, K)
    plot_mesh(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes)
    #check_boundary_conditions(fixed_x_nodes, fixed_xy_nodes, len(mesh_points))
    
    #K_e_list = []

    # Сборка глобальной матрицы жесткости
    K, B_matrices, D = assemble_global_stiffness_matrix(mesh_points, mesh_elements, E, nu)

    #check_local_stiffness_matrices(K_e_list)
    #check_global_stiffness_matrix(K)

    # Применение SW
    F = SW_load(mesh_points, mesh_elements, rho, fixed_xy_nodes, g=9.81)

    # Решение системы
    U = solve_system(K, F, fixed_x_nodes, fixed_xy_nodes)
    # Построение сетки и результатов

    plot_displacement(mesh_points, mesh_elements, U)
    
    print(f'B shape: {B_matrices.shape}')
    print(f'D shape: {D.shape}')

    strains = compute_strains(U, B_matrices)

    stresses = compute_stresses(strains, D)

    # Plot strains
    plot_strains(strains, mesh_points, mesh_elements, title="Strain Visualization", cmap="inferno", show_colorbar=True)

    # Plot stresses
    plot_stresses(stresses, mesh_points, mesh_elements, title="Stress Visualization", cmap="coolwarm", show_colorbar=True)

    plot_x_stress_isobars(stresses, mesh_points, mesh_elements, title="X-Stress Isobar Visualization", cmap="viridis", levels=10)
    plot_y_stress_isobars(stresses, mesh_points, mesh_elements, title="Y-Stress Isobar Visualization", cmap="viridis", levels=10)
    #Sigma = stress_from_strain(B_matrices, D_matrices, U)

    #plot_stress(mesh_points, mesh_elements, Sigma)
