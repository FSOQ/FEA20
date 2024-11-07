import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from utils.plotting import plot_mesh, plot_displacement, plot_strains, plot_stresses, plot_x_stress_isobars, plot_y_stress_isobars
from utils.data_management import save_data_to_csv, load_data_from_csv
from src.model_solver import run_fem_solver
from src.mesh_generation import generate_and_modify_mesh

# Global variable to keep track of the current figure
current_canvas = None
current_fig = None

# Global list to hold materials data (for demo purposes)
materials_data = []


def on_run_create_mesh():
    file_path = './data/generated_mesh.csv'
    generate_and_modify_mesh(file_path, materials_data)


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


    
# Function to show all materials and their properties in a table
def show_existing_materials():
    # Create a new window to show the table
    if not materials_data:
        messagebox.showinfo("No Materials", "No materials data available.")
        return

    table_window = tk.Toplevel()
    table_window.title("Existing Materials")
    table_window.geometry("600x400")

    # Create a Treeview widget to display the materials table
    treeview = ttk.Treeview(table_window, columns=("Material", "E", "nu", "c", "phi", "dilatancy", "rho", "height"), show="headings")

    # Define the columns for the table
    treeview.heading("Material", text="Material")
    treeview.heading("E", text="E")
    treeview.heading("nu", text="nu")
    treeview.heading("c", text="c")
    treeview.heading("phi", text="phi")
    treeview.heading("dilatancy", text="Dilatancy")
    treeview.heading("rho", text="rho")
    treeview.heading("height", text="Height")

    # Set column width
    treeview.column("Material", width=100)
    treeview.column("E", width=100)
    treeview.column("nu", width=100)
    treeview.column("c", width=100)
    treeview.column("phi", width=100)
    treeview.column("dilatancy", width=100)
    treeview.column("rho", width=100)
    treeview.column("height", width=100)

    # Insert existing materials into the table
    for material_name, params in materials_data:
        treeview.insert("", "end", values=(
            material_name, params['E'], params['nu'], params['c'], params['phi'], params['dilatancy'], params['rho'], params['height']
        ))

    treeview.pack(expand=True, fill="both")


# Main Solver import
def on_run_solver():
    run_fem_solver(materials_data)

# Define the functions to plot the mesh, displacement, strains, and stresses
def plot_mesh(mesh_points, mesh_elements, fixed_x_nodes, fixed_xy_nodes):
    # Example function to plot the mesh (replace with actual plotting logic)
    for element in mesh_elements:
        x = mesh_points[element, 0]
        y = mesh_points[element, 1]
        plt.fill(x, y, edgecolor='black', fill=False)

def plot_displacement(mesh_points, mesh_elements, U):
    # Example function to plot the displacement (replace with actual plotting logic)
    for element in mesh_elements:
        x = mesh_points[element, 0]
        y = mesh_points[element, 1]
        plt.fill(x, y, edgecolor='black', fill=False)
    # Add displacement arrows for visualization
    for i, point in enumerate(mesh_points):
        plt.arrow(point[0], point[1], U[i], U[i], head_width=0.05, color='red')

def plot_strains(strains, mesh_points, mesh_elements, title="Strain Visualization", cmap="inferno", show_colorbar=True):
    # Example function to plot strains (replace with actual plotting logic)
    plt.tricontourf(mesh_points[:, 0], mesh_points[:, 1], mesh_elements, strains, cmap=cmap)
    plt.title(title)
    if show_colorbar:
        plt.colorbar()

def plot_stresses(stresses, mesh_points, mesh_elements, title="Stress Visualization", cmap="coolwarm", show_colorbar=True):
    # Example function to plot stresses (replace with actual plotting logic)
    plt.tricontourf(mesh_points[:, 0], mesh_points[:, 1], mesh_elements, stresses, cmap=cmap)
    plt.title(title)
    if show_colorbar:
        plt.colorbar()

def plot_x_stress_isobars(stresses, mesh_points, mesh_elements, title="X-Stress Isobar Visualization", cmap="viridis", levels=10):
    # Example function to plot X-stress isobars (replace with actual plotting logic)
    plt.contourf(mesh_points[:, 0], mesh_points[:, 1], stresses, levels=levels, cmap=cmap)
    plt.title(title)

def plot_y_stress_isobars(stresses, mesh_points, mesh_elements, title="Y-Stress Isobar Visualization", cmap="viridis", levels=10):
    # Example function to plot Y-stress isobars (replace with actual plotting logic)
    plt.contourf(mesh_points[:, 0], mesh_points[:, 1], stresses, levels=levels, cmap=cmap)
    plt.title(title)

# Main GUI
def create_gui():
    global current_canvas, current_fig
    # Create the main window
    window = tk.Tk()
    window.title("FEM Solver Interface and Plotting")
    window.geometry("700x600")

    # Create a frame for the buttons
    button_frame = tk.Frame(window)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    # Create a frame for the plot area
    plot_frame = tk.Frame(window)
    plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def save_data_button_callback():
        save_data_to_csv(mesh_points, material_properties, boundary_conditions)

    def load_data_button_callback():
        mesh_points, material_properties, boundary_conditions = load_data_from_csv()
        if mesh_points is not None:
            # Update the GUI or solver with the loaded data
            pass

    # Function to clear existing plots and close the previous figure
    def clear_plot():
        global current_canvas, current_fig
        if current_canvas:
            current_canvas.get_tk_widget().destroy()
            current_canvas = None
        if current_fig:
            plt.close(current_fig)  # Close the figure to release memory
            current_fig = None

    def plot_mesh_callback():
        clear_plot()
        fig, ax = plt.subplots()
        plot_mesh(mesh_points, mesh_elements, fixed_x_nodes=None, fixed_xy_nodes=None)  # Pass appropriate params
        current_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        current_fig = fig

    def plot_displacement_callback():
        clear_plot()
        fig, ax = plt.subplots()
        plot_displacement(mesh_points, mesh_elements, U)
        current_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        current_fig = fig

    def plot_strains_callback():
        clear_plot()
        fig, ax = plt.subplots()
        plot_strains(strains, mesh_points, mesh_elements, title="Strain Visualization", cmap="inferno", show_colorbar=True)
        current_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        current_fig = fig

    def plot_stresses_callback():
        clear_plot()
        fig, ax = plt.subplots()
        plot_stresses(stresses, mesh_points, mesh_elements, title="Stress Visualization", cmap="coolwarm", show_colorbar=True)
        current_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        current_fig = fig

    def plot_x_isobars_callback():
        clear_plot()
        fig, ax = plt.subplots()
        plot_x_stress_isobars(stresses, mesh_points, mesh_elements, title="X-Stress Isobar Visualization", cmap="viridis", levels=10)
        current_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        current_fig = fig

    def plot_y_isobars_callback():
        clear_plot()
        fig, ax = plt.subplots()
        plot_y_stress_isobars(stresses, mesh_points, mesh_elements, title="Y-Stress Isobar Visualization", cmap="viridis", levels=10)
        current_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        current_fig = fig

    # Buttons for various actions
    button_mesh = ttk.Button(button_frame, text="Plot Mesh", command=plot_mesh_callback)
    button_mesh.pack(pady=5)

    button_displacement = ttk.Button(button_frame, text="Plot Displacement", command=plot_displacement_callback)
    button_displacement.pack(pady=5)

    button_strains = ttk.Button(button_frame, text="Plot Strains", command=plot_strains_callback)
    button_strains.pack(pady=5)

    button_stresses = ttk.Button(button_frame, text="Plot Stresses", command=plot_stresses_callback)
    button_stresses.pack(pady=5)

    button_x_isobars = ttk.Button(button_frame, text="Plot X-Stress Isobars", command=plot_x_isobars_callback)
    button_x_isobars.pack(pady=5)

    button_y_isobars = ttk.Button(button_frame, text="Plot Y-Stress Isobars", command=plot_y_isobars_callback)
    button_y_isobars.pack(pady=5)

        # Add buttons for Save and Load
    button_save = ttk.Button(button_frame, text="Save Data", command=save_data_button_callback)
    button_save.pack(pady=5)

    button_load = ttk.Button(button_frame, text="Load Data", command=load_data_button_callback)
    button_load.pack(pady=5)

        # Add button to show existing materials
    button_show_materials = ttk.Button(button_frame, text="View Existing Materials", command=show_existing_materials)
    button_show_materials.pack(pady=5)

    # Add button to input new materials
    button_input_materials = ttk.Button(button_frame, text="Input New Materials", command=input_materials)
    button_input_materials.pack(pady=5)

    button_mesh_generation = ttk.Button(button_frame, text="Mesh generation", command=on_run_create_mesh)
    button_mesh_generation.pack(pady=5)

        # Add a button to run the FEM solver
    button_run_solver = ttk.Button(button_frame, text="Run FEM Solver", command=on_run_solver)
    button_run_solver.pack(pady=10)

    # Start the Tkinter event loop
    window.mainloop()
