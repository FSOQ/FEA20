import meshpy.triangle as triangle
import pandas as pd
import numpy as np



# Define a function to load mesh data from a CSV file
def load_mesh_from_file(file_path):
    # Load data from CSV
    try:
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


    # Convert 'X' and 'Y' columns into a list of tuples for points
    points = list(mesh_df[['X', 'Y']].itertuples(index=False, name=None))

    # Only include valid facets where both 'Node1' and 'Node2' are present (drop rows with NaN)
    valid_facets = mesh_df.dropna(subset=['Node1', 'Node2'])

    # Convert 'Node1' and 'Node2' to a list of tuples for facets, and adjust for 0-based index
    facets = [(int(row[0]), int(row[1])) for row in valid_facets[['Node1', 'Node2']].values]

    return np.array(points), np.array(facets)

#points = [(0, 10), (10, 10), (15, 5), (30, 5), (30, 0), (0, 0)]
#facets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

def refine(vertices, area):
    return area > 0.5

def create_mesh(points, facets):
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    mesh = triangle.build(mesh_info, refinement_func=refine)
    return mesh
