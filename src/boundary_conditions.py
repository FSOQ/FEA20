import numpy as np

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
