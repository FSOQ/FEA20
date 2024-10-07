import numpy as np

def solve_system(K, F, fixed_dof_indices):
    #Getting rows number ([1] for coloumns)
    num_dof = K.shape[0]
        # Граничные условия: объединение узлов с фиксированными перемещениями
    #fixed_dof_indices = fixed_x_nodes + fixed_xy_nodes
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

    U[free_dof_indices] = U_free    

    """"
    for i, dof_index in enumerate(free_dof_indices):
        U[dof_index] = U_free[i, 1]  # Only using the y-component, assuming you want to ignore x    


        # Assign the free displacements to their corresponding DOFs

    for i, dof_index in enumerate(free_dof_indices):
        if dof_index % 2 == 0:
            # x-component (even indices)
            U[dof_index] = U_free[i]  # Assign x-component displacement
        else:
            # y-component (odd indices)
            U[dof_index] = U_free[i]  # Assign y-component displacement

    # Присваиваем фиксированным степеням свободы их известные значения (например, ноль)
    #U[fixed_x_nodes] = 0  # Если узлы зафиксированы по x
    #U[fixed_xy_nodes] = 0  # Если узлы зафиксированы по x и y
    
    for node in fixed_x_nodes:
        U[2 * node] = 0  # Zero displacement in x-direction for fixed x nodes
    for node in fixed_xy_nodes:
        U[2 * node] = 0      # Zero displacement in x-direction for fully fixed nodes
        U[2 * node + 1] = 0  # Zero displacement in y-direction for fully fixed nodes
    """
    print(f'U shape: {U.shape}')
    

    return U
