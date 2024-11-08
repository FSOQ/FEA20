import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from utils.plotting import plot_mesh, plot_displacement, plot_strains, plot_stresses, plot_x_stress_isobars, plot_y_stress_isobars
from utils.data_management import save_data_to_csv, load_data_from_csv
from src.model_solver import run_fem_solver
from src.mesh_generation import generate_and_modify_mesh

# Global variables to store mesh and solver results

current_canvas = None
current_fig = None
mesh_points = None
mesh_elements = None
U = None
strains = None
stresses = None
materials_data = []
material_properties = None  # Global placeholder
boundary_conditions = None  # Global placeholder


def clear_plot():
    global current_canvas, current_fig
    if current_canvas:
        current_canvas.get_tk_widget().destroy()
        current_canvas = None
    if current_fig:
        plt.close(current_fig)
        current_fig = None


def on_run_create_mesh():
    global mesh_points, mesh_elements
    file_path = './data/generated_mesh.csv'
    generate_and_modify_mesh(file_path, materials_data)
    mesh_points, mesh_elements = load_mesh_from_file(file_path)


def on_run_solver():
    global U, strains, stresses
    fixed_x_nodes, fixed_xy_nodes, U, strains, stresses = run_fem_solver(mesh_points, mesh_elements, materials_data)



def input_materials():
    # Create the material input window
    window = tk.Tk()
    window.title("Material Input")
    window.geometry("400x400")

    # List to hold the material entries
    material_entries = []

    # Function to add more material input fields
    def add_material():
        material_frame = ttk.Frame(materials_frame)
        material_frame.pack(fill="x", padx=10, pady=5)

        # Material name input
        material_name_label = ttk.Label(material_frame, text="Material Name:")
        material_name_label.grid(row=0, column=0, padx=5, pady=5)
        material_name_entry = ttk.Entry(material_frame)
        material_name_entry.grid(row=0, column=1, padx=5, pady=5)

        # Material parameters (E, nu, c, phi, dilatancy, rho, height)
        params = ['E', 'nu', 'c', 'phi', 'dilatancy', 'rho', 'height']
        entries = {}
        for i, param in enumerate(params):
            label = ttk.Label(material_frame, text=param + ":")
            label.grid(row=i + 1, column=0, padx=5, pady=5)
            entry = ttk.Entry(material_frame)
            entry.grid(row=i + 1, column=1, padx=5, pady=5)
            entries[param] = entry

        material_entries.append((material_name_entry, entries))

    # Function to collect and process data
    def submit_materials():
        global materials_data

        new_materials_data = []

        # Collect the data for each material
        for material_entry, param_entries in material_entries:
            material_name = material_entry.get()
            params = {}

            # Validate and convert entries to floats
            for param, entry in param_entries.items():
                try:
                    params[param] = float(entry.get())
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {param}!")
                    return

            new_materials_data.append((material_name, params))

        if not new_materials_data:
            messagebox.showerror("Error", "No materials data entered!")
            return

        # Append new materials data to the global materials_data list
        materials_data.extend(new_materials_data)

        # Process the collected materials data
        for material_name, params in new_materials_data:
            print(f"Material: {material_name}")
            print(f"Parameters: {params}")

        messagebox.showinfo("Success", "Materials input collected successfully!")
        window.quit()  # Close the window after submission
        window.destroy()  # Ensure the window is completely destroyed

    # Button to add more material input fields
    add_material_button = ttk.Button(window, text="Add Material", command=add_material)
    add_material_button.pack(pady=10)

    # Frame to hold material input fields
    materials_frame = ttk.Frame(window)
    materials_frame.pack(padx=10, pady=10)

    # Submit button
    submit_button = ttk.Button(window, text="Submit", command=submit_materials)
    submit_button.pack(pady=20)

    # Start the Tkinter event loop
    window.mainloop()

   


def save_data_button_callback():
    if mesh_points is None or materials_data is None:
        messagebox.showerror("Error", "Cannot save. Mesh or materials not generated!")
        return
    save_data_to_csv(mesh_points, material_properties, boundary_conditions)
    messagebox.showinfo("Success", "Data has been saved successfully!")


def load_data_button_callback():
    global mesh_points, material_properties, boundary_conditions
    mesh_points, material_properties, boundary_conditions = load_data_from_csv()
    if mesh_points is not None:
        messagebox.showinfo("Success", "Data has been loaded successfully!")
    else:
        messagebox.showerror("Error", "Failed to load data!")


def plot_mesh_callback():
    if mesh_points is None or mesh_elements is None:
        messagebox.showerror("Error", "Mesh not generated yet!")
        return
    clear_plot()
    fig, ax = plt.subplots()
    plot_mesh(mesh_points, mesh_elements)
    set_plot_to_canvas(fig)


def plot_displacement_callback():
    if U is None:
        messagebox.showerror("Error", "Solver not run yet!")
        return
    clear_plot()
    fig, ax = plt.subplots()
    plot_displacement(mesh_points, mesh_elements, U)
    set_plot_to_canvas(fig)


def plot_strains_callback():
    if strains is None:
        messagebox.showerror("Error", "Solver not run yet!")
        return
    clear_plot()
    fig, ax = plt.subplots()
    plot_strains(strains, mesh_points, mesh_elements)
    set_plot_to_canvas(fig)


def plot_stresses_callback():
    if stresses is None:
        messagebox.showerror("Error", "Solver not run yet!")
        return
    clear_plot()
    fig, ax = plt.subplots()
    plot_stresses(stresses, mesh_points, mesh_elements)
    set_plot_to_canvas(fig)
    current_fig = fig

def clear_plot():
    global current_canvas, current_fig
    if current_canvas:
        current_canvas.get_tk_widget().destroy()
        current_canvas = None
    if current_fig:
        plt.close(current_fig)
        current_fig = None


def create_button(parent, text, command):
    return ttk.Button(parent, text=text, command=command)

def add_new_material_row():
    treeview.insert("", "end", values=("", "", "", "", "", "", "", ""))

def create_grid(tree):
    """Apply styling for grid lines."""
    # Even and odd row configurations for visible 'grid'
    tree.tag_configure('evenrow', background='white')
    tree.tag_configure('oddrow', background='lightblue')

def prevent_edit(event):
    return "break"  # Prevent the default behavior (editing) for this event

def on_cell_single_click(event):
    # Your logic to allow editing on single click
    row_id = treeview.selection()
    column = treeview.identify_column(event.x)
    
    # This function should create an Entry widget and allow editing
    if row_id and column:
        edit_cell(treeview, row_id, column, event)

    def save_edit(event):
        # Save edited value back to Treeview
        new_value = entry.get()
        treeview.set(row_id, column, new_value)

        # Update the materials_data list
        material_name = treeview.set(row_id, "#1")
        if material_name:  # Avoid adding if material name is blank
            # If row already exists, update the existing data
            if len(materials_data) > int(row_id):
                materials_data[int(row_id)] = {
                    "Material": material_name,
                    "E": float(treeview.set(row_id, "#2") or 0),
                    "nu": float(treeview.set(row_id, "#3") or 0),
                    "c": float(treeview.set(row_id, "#4") or 0),
                    "phi": float(treeview.set(row_id, "#5") or 0),
                    "dilatancy": float(treeview.set(row_id, "#6") or 0),
                    "rho": float(treeview.set(row_id, "#7") or 0),
                    "height": float(treeview.set(row_id, "#8") or 0)
                }
            else:
                # Add new material if row does not exist
                materials_data.append({
                    "Material": material_name,
                    "E": float(treeview.set(row_id, "#2") or 0),
                    "nu": float(treeview.set(row_id, "#3") or 0),
                    "c": float(treeview.set(row_id, "#4") or 0),
                    "phi": float(treeview.set(row_id, "#5") or 0),
                    "dilatancy": float(treeview.set(row_id, "#6") or 0),
                    "rho": float(treeview.set(row_id, "#7") or 0),
                    "height": float(treeview.set(row_id, "#8") or 0)
                })

        entry.destroy()

    entry.bind("<Return>", save_edit)
    entry.bind("<FocusOut>", lambda e: entry.destroy())
    entry.focus()

def show_existing_materials():
    for item in treeview.get_children():
        treeview.delete(item)

    for material in materials_data:
        treeview.insert("", "end", values=(
            material["Material"], material['E'], material['nu'], material['c'], material['phi'], material['dilatancy'], material['rho'], material['height']
        ))

# Main GUI
def create_gui():
    global plot_frame, treeview

    # Create the main window
    window = tk.Tk()
    window.title("FEM Solver Interface and Plotting")
    window.geometry("1200x800")

    # Button frame at the top
    button_frame = tk.Frame(window)
    button_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

    # Left section: Material table
    materials_frame = tk.Frame(window)
    materials_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # Right section: Plot frame
    plot_frame = tk.Frame(window)
    plot_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

    # Configure grid layout to expand properly
    window.grid_columnconfigure(1, weight=1)
    window.grid_rowconfigure(1, weight=1)

    # Add the material table on the left
    treeview = ttk.Treeview(materials_frame, columns=("Material", "E", "nu", "c", "phi", "dilatancy", "rho", "height"), show="headings")
    
    # Set up column headers
    for col in treeview["columns"]:
        treeview.heading(col, text=col)
        treeview.column(col, width=100)

    # Add grid styling and populate data
    create_grid(treeview)
    
    # Bindings
    #treeview.bind("<Double-1>", prevent_edit)  # Disable double-click editing
    #treeview.bind("<Return>", prevent_edit)    # Disable Enter key editing
    treeview.bind("<Button-1>", on_cell_single_click)  # Allow single-click editing

    # Pack the treeview to fit the frame
    treeview.pack(expand=True, fill="both")
    
    # Focus on the first row to enable arrow key navigation
    if treeview.get_children():
        treeview.focus(treeview.get_children()[0])

    # Add "Add New Material" button below the materials table
    add_material_button = create_button(materials_frame, "Add New Material", add_new_material_row)
    add_material_button.pack(side=tk.BOTTOM, pady=5)

    # Add buttons in a single row at the top
    create_button(button_frame, "Plot Mesh", plot_mesh_callback).grid(row=0, column=0, padx=5)
    create_button(button_frame, "Plot Displacement", plot_displacement_callback).grid(row=0, column=1, padx=5)
    create_button(button_frame, "Plot Strains", plot_strains_callback).grid(row=0, column=2, padx=5)
    create_button(button_frame, "Plot Stresses", plot_stresses_callback).grid(row=0, column=3, padx=5)
    create_button(button_frame, "Mesh Generation", on_run_create_mesh).grid(row=0, column=4, padx=5)
    create_button(button_frame, "Run FEM Solver", on_run_solver).grid(row=0, column=5, padx=5)
    create_button(button_frame, "Save Data", save_data_button_callback).grid(row=0, column=6, padx=5)
    create_button(button_frame, "Load Data", load_data_button_callback).grid(row=0, column=7, padx=5)

    window.mainloop()