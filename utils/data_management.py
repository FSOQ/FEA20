import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Save data to CSV/Excel with file path selection
def save_data_to_csv(mesh_points, material_properties, boundary_conditions):
    # Open a save file dialog to select the save location and file name
    filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    
    if not filename:
        print("No file selected. Data not saved.")
        return

    mesh_df = pd.DataFrame(mesh_points, columns=["X", "Y"])
    material_df = pd.DataFrame([material_properties])
    boundary_df = pd.DataFrame([boundary_conditions])

    with pd.ExcelWriter(filename) as writer:
        mesh_df.to_excel(writer, sheet_name="Mesh Points", index=False)
        material_df.to_excel(writer, sheet_name="Material Properties", index=False)
        boundary_df.to_excel(writer, sheet_name="Boundary Conditions", index=False)
    print(f"Data saved to {filename}")

# Load data from CSV/Excel with file path selection
def load_data_from_csv():
    # Open a file dialog to select the file to load
    filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])

    if not filename:
        print("No file selected. Data not loaded.")
        return None, None, None

    data = pd.read_excel(filename, sheet_name=None)  # Read all sheets into a dictionary
    mesh_points = data["Mesh Points"].values.tolist()
    material_properties = data["Material Properties"].iloc[0].to_dict()
    boundary_conditions = data["Boundary Conditions"].iloc[0].to_dict()

    print(f"Data loaded from {filename}")
    return mesh_points, material_properties, boundary_conditions
