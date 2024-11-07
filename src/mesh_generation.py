import meshpy.triangle as triangle
import pandas as pd
import numpy as np
import math

def save_mesh_to_file(points, facets, file_path):
    """
    Saves mesh data (points and facets) to a CSV file.
    Points will be stored with X, Y coordinates, and facets will be stored with node indices.
    """

    # Convert meshpy RealArray (points) and IntArray (facets) to numpy arrays
    points = np.array(points)
    facets = np.array(facets)

    # Prepare data to save
    mesh_data = []

    # Add points to the data
    for point in points:
        mesh_data.append(['Point', point[0], point[1], '', '', ''])

    # Add facets to the data
    for facet in facets:
        mesh_data.append(['Facet', '', '', facet[0], facet[1], facet[2]])

    # Create DataFrame from the mesh_data list
    mesh_df = pd.DataFrame(mesh_data, columns=['Type', 'X', 'Y', 'Node1', 'Node2', 'Node3'])

    # Save the mesh data to a CSV file
    mesh_df.to_csv(file_path, sep=';', index=False)
    print(f"Mesh data saved to {file_path}")

def load_mesh_from_file(file_path):
    """
    Loads mesh data (points and facets) from a CSV file.
    It returns points and facets as numpy arrays.
    """

    try:
        # Load data from CSV
        mesh_df = pd.read_csv(file_path, delimiter=';')
    except FileNotFoundError:
        print("The specified file was not found.")
        return None, None  # Return None if the file was not found
    except pd.errors.EmptyDataError:
        print("The file is empty.")
        return None, None  # Return None if the file is empty
    except pd.errors.ParserError:
        print("Error parsing the file.")
        return None, None  # Return None if there is a parsing error

    # Separate points and facets based on 'Type' column
    points = mesh_df[mesh_df['Type'] == 'Point'][['X', 'Y']].values
    facets = mesh_df[mesh_df['Type'] == 'Facet'][['Node1', 'Node2', 'Node3']].values

    return points, facets

def create_layered_mesh(materials_data, width=30):
    """
    Creates points and facets for stacked layers of rectangles based on material heights.
    This output is designed to be compatible with MeshPy's requirements.
    """
    points = []
    facets = []
    current_height = 0

    for i, (material_name, params) in enumerate(materials_data):
        print(f"Material: {material_name}")
        print(f"Parameters: {params}")
        height = params["height"]  # Get the height of the current material layer

        # Define the four corner points of the rectangle
        layer_points = [
            [0, current_height],  # Bottom-left
            [width, current_height],  # Bottom-right
            [width, current_height + height],  # Top-right
            [0, current_height + height]  # Top-left
        ]
        
        start_index = len(points)  # Get the starting index for the current layer's points

        # Add the new layer points to the points list
        points.extend(layer_points)

        # Define the facets (edges) for this rectangle layer
        # Each facet connects consecutive points and closes the rectangle
        layer_facets = [
            [start_index, start_index + 1],  # Bottom edge
            [start_index + 1, start_index + 2],  # Right edge
            [start_index + 2, start_index + 3],  # Top edge
            [start_index + 3, start_index]  # Left edge
        ]

        # Add the layer facets to the facets list
        facets.extend(layer_facets)

        # Update the current height to move to the next layer
        current_height += height

    return np.array(points), np.array(facets)

def modify_mesh_for_slope(mesh_points, mesh_elements, width=30, slope_height=5, slope_angle=45):
    """
    Modify the existing mesh by removing the top center region based on the slope height and angle.

    Args:
    mesh_points (np.array): Existing mesh points (2D array of shape Nx2).
    mesh_elements (np.array): Existing mesh elements (2D array of shape Mx3 or Mx4).
    width (float): The width of the top rectangle.
    slope_height (float): The height of the slope.
    slope_angle (float): The angle of the slope in degrees.

    Returns:
    new_points (np.array): Modified points after cutting out the slope region.
    new_elements (np.array): Modified elements after removing parts within the slope.
    """

    # Find the center of the top rectangle
    top_center_x = width / 2
    top_center_y = np.max(mesh_points[:, 1])  # Y-coordinate of the topmost points

    # Calculate the slope endpoints based on the angle and height
    angle_rad = math.radians(slope_angle)
    delta_x = slope_height / math.tan(angle_rad)

    slope_left_x = top_center_x - delta_x
    slope_right_x = top_center_x + delta_x
    slope_bottom_y = top_center_y - slope_height

    # Define a region to be removed (above the slope line)
    def is_above_slope(point):
        x, y = point
        if x < slope_left_x or x > slope_right_x:
            return False
        slope_y_at_x = slope_bottom_y + (abs(x - top_center_x) / delta_x) * slope_height
        return y > slope_y_at_x

    # Filter points and elements to remove those above the slope
    keep_points = []
    point_map = {}  # A map from old point indices to new ones
    new_index = 0

    for i, point in enumerate(mesh_points):
        if not is_above_slope(point):
            keep_points.append(point)
            point_map[i] = new_index
            new_index += 1

    keep_points = np.array(keep_points)

    # Keep only elements that consist of remaining points
    new_elements = []
    for element in mesh_elements:
        if all(point in point_map for point in element):
            new_elements.append([point_map[point] for point in element])

    new_elements = np.array(new_elements)

    # Optionally: add new points and facets along the slope edges if needed
    # For example, add the slope_left and slope_right points at the appropriate positions.

    return keep_points, new_elements

def refine(vertices, area):
    return area > 0.5

def create_mesh(points, facets):
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh = triangle.build(mesh_info, refinement_func=refine)
    return mesh

def generate_and_modify_mesh(file_path, materials_data):
    # Step 1: Create layered mesh (30xheight for each layer) 
    width = 30  # Width of each layer
    points, elements = create_layered_mesh(materials_data, width)

    # Step 2: Modify the mesh for the slope
    slope_height = 5  # Example slope height
    slope_angle = 30  # Example slope angle in degrees
    modified_points, modified_elements = modify_mesh_for_slope(points, elements, width, slope_height, slope_angle)
 
    # Create the mesh (this is decoupled from the solver now)
    mesh = create_mesh(modified_points, modified_elements)

    # Save the mesh points and facets to a CSV file
    save_mesh_to_file(mesh.points, mesh.elements, file_path)

    print("Mesh generation and modification complete!")



    
