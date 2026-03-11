"""
Graphical User Interface Module.

This module provides a Tkinter-based front-end for the SPICE simulator. It allows
users to load netlists, run simulations asynchronously, and visualize the output
using embedded Matplotlib figures.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading

class CircuitSimulatorGUI:
    """The main GUI application for the Python SPICE Simulator.

    Attributes:
        root (tk.Tk): The root Tkinter window.
        run_simulation_core (callable): The backend entry point function from main.py.
        circuit (Circuit): The active parsed circuit object.
        result (SimulationResult): The active simulation data vault.
    """

    def __init__(self, root, simulation_callback):
        """Initializes the GUI layout and control bindings.

        Args:
            root (tk.Tk): The parent Tkinter window.
            simulation_callback (callable): The function that executes the simulation.
        """
        self.root = root
        self.root.title("Python SPICE Simulator")
        self.root.geometry("1200x800")
        
        # Link to the main.py calculation engine
        self.run_simulation_core = simulation_callback

        # Storage for simulation objects
        self.circuit = None 
        self.result = None

        # --- Top Control Panel ---
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_load = tk.Button(control_frame, text="Load Netlist", command=self.load_file, width=15)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.lbl_file = tk.Label(control_frame, text="No file selected", fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_run = tk.Button(control_frame, text="Run Simulation", command=self.run_simulation, width=15, bg="#90ee90", state=tk.DISABLED)
        self.btn_run.pack(side=tk.RIGHT, padx=5)

        # --- Main Layout: Paned Window ---
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
        self.node_listbox = tk.Listbox(selector_frame, selectmode=tk.MULTIPLE, font=("Arial", 10), height=15)
        self.node_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.node_listbox.bind('<<ListboxSelect>>', self.on_selection_change)

        tk.Label(selector_frame, text="Sensitivity Target:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        self.sens_cb = ttk.Combobox(selector_frame, state="readonly")
        self.sens_cb.pack(fill=tk.X, padx=5, pady=5)
        self.sens_cb.bind("<<ComboboxSelected>>", self.on_selection_change)

        # --- Tab 2: Nodal Analysis Data ---
        self.tab_data = tk.Frame(self.tabs)
        self.tabs.add(self.tab_data, text="Nodal Analysis Data")
        self.data_text = tk.Text(self.tab_data, font=("Courier", 10))
        self.data_text.pack(fill=tk.BOTH, expand=True)

        self.current_file_path = None
        self.node_names_cache = [] # Maps listbox indices directly to node string names

    def load_file(self):
        """Opens a file dialog to load a SPICE netlist text file."""
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.current_file_path = file_path
            self.lbl_file.config(text=file_path.split("/")[-1], fg="black")
            self.btn_run.config(state=tk.NORMAL)
            with open(file_path, "r") as f:
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, f.read())

    def run_simulation(self):
        """Saves live edits to the text file and starts the simulation thread."""
        if self.current_file_path:
            with open(self.current_file_path, "w") as f:
                f.write(self.text_area.get(1.0, tk.END))
        else:
            messagebox.showerror("Error", "No file loaded.")
            return

        self.btn_run.config(state=tk.DISABLED, text="Simulating...", bg="#cccccc")
        
        # Start the heavy math in a background thread to keep GUI responsive
        threading.Thread(target=self._simulation_thread, daemon=True).start()

    def _simulation_thread(self):
        """Executes the core numerical engine off the main GUI thread."""
        try:
            # We now unpack the Circuit and SimulationResult OOP objects!
            self.circuit, self.result = self.run_simulation_core(
                self.current_file_path, 
                sensitivity=True
            )
            # Safely push results back to the GUI thread
            self.root.after(0, self._simulation_complete)
        except Exception as e:
            self.root.after(0, lambda e=e: self._simulation_error(e))

    def _simulation_error(self, error):
        """Restores the UI and displays exception tracebacks."""
        self.btn_run.config(state=tk.NORMAL, text="Run Simulation", bg="#90ee90")
        messagebox.showerror("Simulation Error", f"Simulation failed:\n\n{str(error)}")

    def _simulation_complete(self):
        """Populates the UI listboxes and triggers plot updates."""
        self.btn_run.config(state=tk.NORMAL, text="Run Simulation", bg="#90ee90")
        
        if not self.result:
            return

        # 1. Populate Node Listbox
        self.node_listbox.delete(0, tk.END)
        self.node_names_cache = []
        
        # Sort nodes alphabetically for consistent display
        sorted_nodes = sorted(self.result.node_map.keys(), key=str)
        mna_prefixes = ('V', 'L', 'E', 'H', 'F')

        for node_name in sorted_nodes:
            self.node_names_cache.append(node_name)
            
            # Format nicely: V(out) vs I(V1)
            if str(node_name).upper().startswith(mna_prefixes):
                display_name = f"I({node_name})"
            else:
                display_name = f"V({node_name})"
                
            self.node_listbox.insert(tk.END, display_name)
            
            # Auto-select voltage nodes by default
            if not str(node_name).upper().startswith(mna_prefixes):
                self.node_listbox.selection_set(tk.END)

        # 2. Populate Sensitivity Component Dropdown using fast OOP dictionary
        self.sens_cb['values'] = ["None"] + list(self.circuit.components_dict.keys())
        self.sens_cb.current(0)

        # 3. Route to correct tab based on analysis type
        if self.result.type == ".OP":
            self.update_data_tab_op()
            self.tabs.select(self.tab_data)
        else:
            self.update_plot()
            self.tabs.select(self.tab_plot)

    def on_selection_change(self, event):
        """Callback for listbox and combobox selections."""
        self.update_plot()

    def update_data_tab_op(self):
        """Displays pure DC Operating Point text."""
        self.data_text.delete(1.0, tk.END)
        
        res_str = "--- DC Operating Point ---\n"
        
        # Use our clean OOP getter!
        for node in self.node_names_cache:
            val = self.result.get_voltage(node)
            unit = "A" if str(node).upper().startswith(('V', 'L')) else "V"
            res_str += f"{str(node):<10} | {val:+.6f} {unit}\n"
            
            # Check for OP sensitivities
            calc_params = self.result.get_sensitivity_parameters(node)
            if calc_params:
                res_str += "  Sensitivities:\n"
                for comp in calc_params:
                    sens_val = self.result.get_sensitivity(node, comp)
                    res_str += f"    -> d({node})/d({comp}) = {sens_val:+.6e}\n"
                res_str += "\n"
                
        self.data_text.insert(tk.END, res_str)

    def update_plot(self):
        """Dynamically draws Bode, Transient, DC Sweep, and Sensitivity plots."""
        if not self.result: return

        self.fig.clf() 
        
        selected_indices = self.node_listbox.curselection()
        if not selected_indices:
            self.canvas.draw()
            return

        x_axis = self.result.sweep_axis
        target_comp = self.sens_cb.get()

        # --- AC / BODE PLOT ---
        if self.result.type == ".AC":
            show_sens = target_comp != "None"
            
            if show_sens:
                ax1 = self.fig.add_subplot(311)
                ax2 = self.fig.add_subplot(312, sharex=ax1)
                ax3 = self.fig.add_subplot(313, sharex=ax1)
            else:
                ax1 = self.fig.add_subplot(211)
                ax2 = self.fig.add_subplot(212, sharex=ax1)
            
            for listbox_idx in selected_indices:
                node_name = self.node_names_cache[listbox_idx]
                label = self.node_listbox.get(listbox_idx)
                
                # Fetch pristine complex arrays via OOP method
                v = self.result.get_voltage(node_name)
                mag_db = 20 * np.log10(np.where(np.abs(v) == 0, 1e-12, np.abs(v)))
                phase = np.angle(v, deg=True)
                
                ax1.semilogx(x_axis, mag_db, label=label, lw=2)
                ax2.semilogx(x_axis, phase, label=label, lw=2)
                
                # Plot Sensitivity if requested
                if show_sens and target_comp in self.result.get_sensitivity_parameters(node_name):
                    raw_sens = self.result.get_sensitivity(node_name, target_comp)
                    ax3.semilogx(x_axis, np.abs(raw_sens), label=f"d({label})/d{target_comp}", lw=2, linestyle='--')

            ax1.set_ylabel("Magnitude (dB)")
            ax1.grid(True, which="both", ls="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=8)
            
            ax2.set_ylabel("Phase (deg)")
            ax2.grid(True, which="both", ls="--", alpha=0.5)
            
            if show_sens:
                ax3.set_ylabel(f"|Sens| w.r.t {target_comp}")
                ax3.set_xlabel("Frequency (Hz)")
                ax3.grid(True, which="both", ls="--", alpha=0.5)
                ax3.legend(loc="upper right", fontsize=8)
            else:
                ax2.set_xlabel("Frequency (Hz)")

        # --- TRANSIENT & DC SWEEP PLOT ---
        elif self.result.type in [".TRAN", ".DC"]:
            show_sens = target_comp != "None"
            
            if show_sens:
                ax1 = self.fig.add_subplot(211)
                ax2 = self.fig.add_subplot(212, sharex=ax1)
            else:
                ax1 = self.fig.add_subplot(111)
            
            for listbox_idx in selected_indices:
                node_name = self.node_names_cache[listbox_idx]
                label = self.node_listbox.get(listbox_idx)
                
                # Automatically strips complex artifacts based on result.type
                v = self.result.get_voltage(node_name)
                ax1.plot(x_axis, v, label=label, lw=2)
                
                if show_sens and target_comp in self.result.get_sensitivity_parameters(node_name):
                    # Fetch the raw time-series array via the Data Vault
                    sens_val = self.result.get_sensitivity(node_name, target_comp, output_format="series")
                    ax2.plot(x_axis, sens_val, label=f"d({label})/d{target_comp}", lw=2, linestyle='--')

            ax1.set_ylabel("Voltage (V) / Current (A)")
            ax1.grid(True, ls="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=8)
            
            if show_sens:
                ax2.set_ylabel(f"Sensitivity w.r.t {target_comp}")
                ax2.set_xlabel("Time (s)" if self.result.type == ".TRAN" else "Sweep Voltage (V)")
                ax2.grid(True, ls="--", alpha=0.5)
                ax2.legend(loc="upper right", fontsize=8)
            else:
                ax1.set_xlabel("Time (s)" if self.result.type == ".TRAN" else "Sweep Voltage (V)")

        self.fig.tight_layout()
        self.canvas.draw()
