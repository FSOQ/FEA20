import numpy as np
from mesh_generation import create_mesh, load_mesh_from_file
from boundary_conditions import apply_boundary_conditions
from Data_process import assemble_global_stiffness_matrix, compute_strains, compute_stresses
from Pre_process import SW_load
from solver import solve_system
from tkinter import messagebox
#from tests.test_boundary_conditions import check_boundary_conditions

def run_fem_solver():
    try:
        # Путь к файлу с точками
        points, facets = load_mesh_from_file('./data/mesh_file.csv')
        # Путь к файлу с материалами
        material_csv_path = "./data/material_properties.csv"
        # Загрузка свойств материала
        #materials_df = load_material_properties(material_csv_path)
        # Пример выбора материала по имени
        #material_name = "Soil_Mat_1"  # Заменить на нужное имя материала
        #E, nu, c, phi, dilatancy, rho = get_material_properties(materials_df, material_name)
        # Extract the first material from the materials_data list
        material_name, params = materials_data[0]

        # Unpack the material properties from the params dictionary
        E = params["E"]
        nu = params["nu"]
        c = params["c"]
        phi = params["phi"]
        dilatancy = params["dilatancy"]
        rho = params["rho"]

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
        fixed_x_nodes, fixed_xy_nodes, free_dof_indices, fixed_dof_indices = apply_boundary_conditions(mesh_points)
        #Вывод матриц
        #display_fem_results(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes, K_e_list, K)
        #check_boundary_conditions(fixed_x_nodes, fixed_xy_nodes, len(mesh_points))
        
        #K_e_list = []

        # Сборка глобальной матрицы жесткости
        K, B_matrices, D = assemble_global_stiffness_matrix(mesh_points, mesh_elements, E, nu)
        #check_local_stiffness_matrices(K_e_list)
        #check_global_stiffness_matrix(K)

        # Применение SW
        F = SW_load(mesh_points, mesh_elements, rho, fixed_xy_nodes, g=9.81)
        # Решение системы
        U = solve_system(K, F, fixed_dof_indices)
        # Построение сетки и результатов
    
        print(f'B shape: {B_matrices.shape}')
        print(f'D shape: {D.shape}')

        strains = compute_strains(U, B_matrices, mesh_elements)

        stresses = compute_stresses(strains, D)
        
        print(f'Strains shape: {strains.shape}')
        print(f'Stresses shape: {stresses.shape}')
       
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during FEM calculation: {str(e)}")
