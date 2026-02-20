import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

class CircuitSimulatorGUI:
    def __init__(self, root, simulation_callback):
        self.root = root
        self.root.title("Python SPICE Simulator")
        self.root.geometry("1200x800")
        
        # Link to the main.py calculation engine
        self.run_simulation_core = simulation_callback

        # Storage for simulation data
        self.sim_data = None 

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

        # --- Tab 1: Waveform Plot ---
        self.tab_plot = tk.Frame(self.tabs)
        self.tabs.add(self.tab_plot, text="Waveform Plot")
        
        plot_pane = tk.PanedWindow(self.tab_plot, orient=tk.HORIZONTAL)
        plot_pane.pack(fill=tk.BOTH, expand=True)

        plot_frame = tk.Frame(plot_pane)
        plot_pane.add(plot_frame, stretch="always")

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        selector_frame = tk.Frame(plot_pane, width=200, bg="#f0f0f0")
        plot_pane.add(selector_frame, stretch="never")
        tk.Label(selector_frame, text="Visible Nodes:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=5)
        
        self.node_listbox = tk.Listbox(selector_frame, selectmode=tk.MULTIPLE, font=("Arial", 10), height=20)
        self.node_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.node_listbox.bind('<<ListboxSelect>>', self.on_node_selection_change)

        # --- Tab 2: Nodal Analysis Data ---
        self.tab_data = tk.Frame(self.tabs)
        self.tabs.add(self.tab_data, text="Nodal Analysis Data")
        self.data_text = tk.Text(self.tab_data, font=("Courier", 10))
        self.data_text.pack(fill=tk.BOTH, expand=True)

        # --- Tab 3: Schematic Editor (Placeholder) ---
        self.tab_schematic = tk.Frame(self.tabs)
        self.tabs.add(self.tab_schematic, text="Schematic Editor")
        self.schematic_canvas = tk.Canvas(self.tab_schematic, bg="white")
        self.schematic_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.schematic_canvas.create_text(
            400, 300, 
            text="Interactive Circuit Drawing\nComing Soon!", 
            font=("Arial", 24, "bold"), fill="#aaaaaa", justify=tk.CENTER
        )

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
            # Call the main engine!
            self.sim_data = self.run_simulation_core(
                self.current_file_path, 
                sensitivity_post=False # Set True if you want to enable sensitivity in GUI
            )
            
            node_map = self.sim_data["node_map"]
            analyses = self.sim_data["analyses"]
            
            # Populate Node Listbox
            self.node_listbox.delete(0, tk.END)
            idx_to_node = {v: k for k, v in node_map.items()}
            sorted_indices = sorted(idx_to_node.keys(), key=lambda i: str(idx_to_node[i]))

            for i in sorted_indices:
                node_name = idx_to_node[i]
                display_name = f"V({node_name})" if isinstance(node_name, int) else f"I({node_name})"
                self.node_listbox.insert(tk.END, display_name)
                
                # Default Select Voltage Nodes
                if isinstance(node_name, int):
                    self.node_listbox.selection_set(tk.END)

            # Route to correct tab
            if ".op" in analyses and ".TRAN" not in analyses and ".ac" not in analyses:
                self.update_data_tab_op()
                self.tabs.select(self.tab_data)
            else:
                self.update_plot()
                self.tabs.select(self.tab_plot)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_node_selection_change(self, event):
        self.update_plot()

    def update_data_tab_op(self):
        """Displays purely DC Operating point text."""
        self.data_text.delete(1.0, tk.END)
        VI = self.sim_data["VI"]
        node_map = self.sim_data["node_map"]
        idx_to_node = {v: k for k, v in node_map.items()}
        
        res_str = "--- DC Operating Point ---\n"
        for idx, val in enumerate(VI):
            node = idx_to_node.get(idx, f"Idx{idx}")
            res_str += f"{str(node):<10} | {val:.6f} V/A\n"
        self.data_text.insert(tk.END, res_str)

    def update_plot(self):
        """Dynamically draws a Bode Plot or Transient Plot."""
        if self.sim_data is None or self.sim_data["VI"] is None: return

        self.fig.clf() # Clear entire figure to rebuild axes
        
        selected_indices = self.node_listbox.curselection()
        if not selected_indices:
            self.canvas.draw()
            return

        x_axis = self.sim_data["x_axis"]
        VI = self.sim_data["VI"]
        analyses = self.sim_data["analyses"]
        node_map = self.sim_data["node_map"]
        
        idx_to_node = {v: k for k, v in node_map.items()}
        sorted_sim_indices = sorted(idx_to_node.keys(), key=lambda i: str(idx_to_node[i]))

        # --- AC / BODE PLOT ---
        if ".ac" in analyses:
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212, sharex=ax1)
            
            for listbox_idx in selected_indices:
                sim_idx = sorted_sim_indices[listbox_idx]
                label = self.node_listbox.get(listbox_idx)
                
                v = VI[:, sim_idx]
                mag = np.abs(v)
                mag_db = 20 * np.log10(np.where(mag == 0, 1e-12, mag))
                phase = np.angle(v, deg=True)
                
                ax1.semilogx(x_axis, mag_db, label=label, lw=2)
                ax2.semilogx(x_axis, phase, label=label, lw=2)

            ax1.set_ylabel("Magnitude (dB)")
            ax1.grid(True, which="both", ls="--", alpha=0.5)
            ax1.legend()
            
            ax2.set_ylabel("Phase (deg)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.grid(True, which="both", ls="--", alpha=0.5)

        # --- TRANSIENT PLOT ---
        elif ".TRAN" in analyses:
            ax = self.fig.add_subplot(111)
            
            for listbox_idx in selected_indices:
                sim_idx = sorted_sim_indices[listbox_idx]
                label = self.node_listbox.get(listbox_idx)
                
                # Take real part to strip any complex artifacts
                ax.plot(x_axis, np.real(VI[:, sim_idx]), label=label, lw=2)

            ax.set_ylabel("Voltage (V) / Current (A)")
            ax.set_xlabel("Time (s)")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()

        self.fig.tight_layout()
        self.canvas.draw()
