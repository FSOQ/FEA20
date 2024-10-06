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
