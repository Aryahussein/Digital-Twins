import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

# Imports from your modules
from txt2dictionary import parse_netlist
from solver import build_node_index, solve_sparse
from assembleYmatrix import generate_basis_matrix, stamp_nonlinear_components, NONLINEAR_DISPATCH
from sources import evaluate_all_time_sources

# --- SIMULATION ENGINE (Unchanged) ---
def run_newton_raphson(components, node_map, total_dim, dt=None, v_prev=None, sources_prev=None, mode="TRAN"):
    MAX_ITER = 50
    TOLERANCE = 1e-6
    
    Y_base, sources_base = generate_basis_matrix(components, node_map, total_dim, dt, v_prev, sources_prev, mode)

    if v_prev is not None:
        v_guess = v_prev.copy()
    else:
        v_guess = np.zeros(total_dim)

    has_nonlinear = any(name[0] in NONLINEAR_DISPATCH for name in components)
    iterations = MAX_ITER if has_nonlinear else 1
    
    for i in range(iterations):
        Y_iter = Y_base.copy()
        sources_iter = sources_base.copy()
        
        if has_nonlinear:
            stamp_nonlinear_components(Y_iter, sources_iter, components, node_map, v_guess)
        
        try:
            v_new = solve_sparse(Y_iter.tocsc(), sources_iter)
        except RuntimeError:
            return None 

        if not has_nonlinear:
            return v_new
            
        if np.max(np.abs(v_new - v_guess)) < TOLERANCE:
            return v_new
            
        v_guess = v_new
        
    return v_guess 

def run_dc_analysis(components):
    node_map, total_dim = build_node_index(components)
    comp_t0 = evaluate_all_time_sources(components, 0.0)
    v_dc = run_newton_raphson(comp_t0, node_map, total_dim, mode="DC")
    return node_map, v_dc

def run_transient_analysis(components, t_stop, dt):
    node_map, total_dim = build_node_index(components)
    num_steps = int(t_stop / dt)
    
    time = np.linspace(0, t_stop, num_steps)
    results = np.zeros((num_steps, total_dim))
    
    v_prev = np.zeros(total_dim)
    sources_prev = np.zeros(total_dim)
    
    for step in range(num_steps):
        t = step * dt
        comp_t = evaluate_all_time_sources(components, t)
        
        v_new = run_newton_raphson(comp_t, node_map, total_dim, dt, v_prev, sources_prev, mode="TRAN")
        
        if v_new is None:
            break
            
        results[step, :] = v_new
        v_prev = v_new
        
        _, s_p = generate_basis_matrix(comp_t, node_map, total_dim, dt, v_prev, sources_prev, mode="TRAN")
        sources_prev = s_p

    return time, results, node_map

# --- GUI CLASS ---

class CircuitSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Python SPICE Simulator")
        self.root.geometry("1200x700")

        # Storage for simulation data (so we can re-plot without re-simulating)
        self.sim_time = None
        self.sim_results = None
        self.sim_node_map = None

        # Top Control Panel
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_load = tk.Button(control_frame, text="Load Netlist", command=self.load_file, width=15)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.lbl_file = tk.Label(control_frame, text="No file selected", fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_run = tk.Button(control_frame, text="Run Simulation", command=self.run_simulation, width=15, bg="#90ee90", state=tk.DISABLED)
        self.btn_run.pack(side=tk.RIGHT, padx=5)

        # Main Layout: Paned Window
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 1. Left: Netlist Text Area
        self.text_area = tk.Text(main_pane, width=30, font=("Courier", 10))
        main_pane.add(self.text_area)

        # 2. Right: Tabs
        self.tabs = ttk.Notebook(main_pane)
        main_pane.add(self.tabs)

        # --- Tab 1: Waveform Plot (Split into Plot + Node List) ---
        self.tab_plot = tk.Frame(self.tabs)
        self.tabs.add(self.tab_plot, text="Waveform Plot")
        
        # Internal PanedWindow for Plot Tab
        plot_pane = tk.PanedWindow(self.tab_plot, orient=tk.HORIZONTAL)
        plot_pane.pack(fill=tk.BOTH, expand=True)

        # Plot Area (Left side of tab)
        plot_frame = tk.Frame(plot_pane)
        plot_pane.add(plot_frame, stretch="always")

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        # Node Selector (Right side of tab)
        selector_frame = tk.Frame(plot_pane, width=200, bg="#f0f0f0")
        plot_pane.add(selector_frame, stretch="never")

        tk.Label(selector_frame, text="Visible Nodes:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Label(selector_frame, text="(Ctrl+Click to select multiple)", bg="#f0f0f0", font=("Arial", 8)).pack(pady=0)

        # Listbox for nodes
        self.node_listbox = tk.Listbox(selector_frame, selectmode=tk.MULTIPLE, font=("Arial", 10), height=20)
        self.node_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.node_listbox.bind('<<ListboxSelect>>', self.on_node_selection_change)

        # --- Tab 2: Nodal Analysis Data ---
        self.tab_data = tk.Frame(self.tabs)
        self.tabs.add(self.tab_data, text="Nodal Analysis Data")
        
        self.data_text = tk.Text(self.tab_data, font=("Courier", 10))
        self.data_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.tab_data, command=self.data_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_text.config(yscrollcommand=scrollbar.set)

        self.current_file_path = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.current_file_path = file_path
            self.lbl_file.config(text=file_path.split("/")[-1], fg="black")
            self.btn_run.config(state=tk.NORMAL)
            with open(file_path, "r") as f:
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, f.read())

    def run_simulation(self):
        try:
            components, analyses = parse_netlist(self.current_file_path)
            self.ax.clear()
            self.data_text.delete(1.0, tk.END)

            # MODE 1: DC OP
            if ".op" in analyses and ".tran" not in analyses:
                node_map, v_dc = run_dc_analysis(components)
                # Display text results only
                res_str = "--- DC Operating Point ---\n"
                idx_to_node = {v: k for k, v in node_map.items()}
                for idx, val in enumerate(v_dc):
                    node = idx_to_node.get(idx, f"Idx{idx}")
                    res_str += f"{node:<10} | {val:.6f} V\n"
                self.data_text.insert(tk.END, res_str)
                self.tabs.select(self.tab_data)
                return

            # MODE 2: TRANSIENT
            t_stop = 1e-3
            dt = 1e-6
            if ".tran" in analyses:
                t_stop = analyses[".tran"]["stop_time"]
                dt = analyses[".tran"]["max_timestep"]
            
            # Run Sim
            self.sim_time, self.sim_results, self.sim_node_map = run_transient_analysis(components, t_stop, dt)
            
            # Populate Node Selector List
            self.node_listbox.delete(0, tk.END)
            idx_to_node = {v: k for k, v in self.sim_node_map.items()}
            
            # Sort by node name for neatness
            sorted_indices = sorted(idx_to_node.keys(), key=lambda i: str(idx_to_node[i]))

            for i in sorted_indices:
                node_name = idx_to_node[i]
                # Filter: Don't show internal Branch Currents (L, V) unless requested
                # Simple check: if it's an integer node, show it.
                display_name = f"V({node_name})" if isinstance(node_name, int) else f"I({node_name})"
                self.node_listbox.insert(tk.END, display_name)
                
            # Default: Select All Voltage Nodes
            for i in range(self.node_listbox.size()):
                if self.node_listbox.get(i).startswith("V("):
                    self.node_listbox.selection_set(i)
            
            # Plot with default selection
            self.update_plot()
            self.tabs.select(self.tab_plot)

            # Show Data Table (First 100 points)
            res_str = f"{'Time':<12} | " + " | ".join([f"{self.node_listbox.get(i)}" for i in range(len(sorted_indices))]) + "\n"
            limit = min(100, len(self.sim_time))
            for i in range(limit):
                row = f"{self.sim_time[i]*1000:<12.3f} | " + " | ".join([f"{val:8.4f}" for val in self.sim_results[i]])
                res_str += row + "\n"
            self.data_text.insert(tk.END, res_str)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise e

    def on_node_selection_change(self, event):
        """Callback when user clicks the listbox."""
        self.update_plot()

    def update_plot(self):
        """Redraws the plot based on selected listbox items."""
        if self.sim_time is None: return

        self.ax.clear()
        
        # Get indices of selected items in the listbox
        selected_indices = self.node_listbox.curselection()
        
        # We need to map listbox index -> simulation matrix index
        # Since we populated the listbox using sorted_indices, we must reconstruct that map
        idx_to_node = {v: k for k, v in self.sim_node_map.items()}
        sorted_sim_indices = sorted(idx_to_node.keys(), key=lambda i: str(idx_to_node[i]))

        for listbox_idx in selected_indices:
            sim_idx = sorted_sim_indices[listbox_idx]
            label = self.node_listbox.get(listbox_idx)
            
            # Plot
            self.ax.plot(self.sim_time * 1000, self.sim_results[:, sim_idx], label=label)

        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Voltage (V) / Current (A)")
        self.ax.grid(True)
        if selected_indices:
            self.ax.legend()
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitSimulatorGUI(root)
    root.mainloop()