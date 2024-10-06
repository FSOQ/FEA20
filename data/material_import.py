import pandas as pd
# Функция для чтения материала
def load_material_properties(csv_path):
    try:
        materials_df = pd.read_csv(csv_path, delimiter=';')
        # Normalize column names
        materials_df.columns = materials_df.columns.str.strip().str.lower()  # Strip whitespace and convert to lowercase
        print("Material Properties Loaded:")
        print(materials_df.columns.tolist())  # Print column names for verification
        return materials_df
    except FileNotFoundError:
        print("The specified file was not found.")
    except pd.errors.EmptyDataError:
        print("The file is empty.")
    except pd.errors.ParserError:
        print("Error parsing the file.")

# Извлечение свойств материала по имени
def get_material_properties(materials_df, material_name):
    material = materials_df[materials_df['material_name'] == material_name].iloc[0]
    E = material['young_modulus']
    nu = material['poisson_ratio']
    rho = material['density']
    c = material['cohesion']
    phi = material['friction_angle']
    dilatancy = material['dilatancy_angle']
    return E, nu, rho, c, phi, dilatancy

#Check git