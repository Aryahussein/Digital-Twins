"""
Graphical User Interface Module.

This module provides a Tkinter-based front-end for the SPICE simulator. It allows
users to load netlists, run simulations asynchronously, and visualize the output
using embedded Matplotlib figures. For transient analysis, the user can choose
between Backward Euler (BE) and Trapezoidal (TR) integration methods, with TR
as the default. A "Compare BE vs TR" mode runs both methods and overlays the
sensitivity differences.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import shutil
import os


class CircuitSimulatorGUI:
    """The main GUI application for the Python SPICE Simulator."""

    def __init__(self, root, simulation_callback):
        self.root = root
        self.root.title("Python SPICE Simulator")
        self.root.geometry("1200x800")
        
        self.run_simulation_core = simulation_callback

        self.circuit = None 
        self.result = None
        self.result_compare = None  # Stores the comparison method result

        # --- Top Control Panel ---
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_load = tk.Button(control_frame, text="Load Netlist", command=self.load_file, width=15)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.lbl_file = tk.Label(control_frame, text="No file selected", fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_run = tk.Button(control_frame, text="Run Simulation", command=self.run_simulation, width=15, bg="#90ee90", state=tk.DISABLED)
        self.btn_run.pack(side=tk.RIGHT, padx=5)

        # Integration method selector
        method_frame = tk.Frame(control_frame)
        method_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Label(method_frame, text="Integration:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="TR")
        self.method_combo = ttk.Combobox(
            method_frame, textvariable=self.method_var, 
            values=["TR", "BE", "Compare BE vs TR"],
            state="readonly", width=16
        )
        self.method_combo.pack(side=tk.LEFT, padx=4)


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
        self.node_names_cache = []

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
        """Saves live edits (with backup) and starts the simulation thread."""
        if not self.current_file_path:
            messagebox.showerror("Error", "No file loaded.")
            return

        backup_path = self.current_file_path + ".bak"
        if os.path.exists(self.current_file_path):
            try:
                shutil.copy2(self.current_file_path, backup_path)
            except OSError:
                pass

        with open(self.current_file_path, "w") as f:
            f.write(self.text_area.get(1.0, tk.END))

        self.btn_run.config(state=tk.DISABLED, text="Simulating...", bg="#cccccc")
        
        threading.Thread(target=self._simulation_thread, daemon=True).start()

    def _get_method_choice(self):
        """Returns the selected integration method(s)."""
        choice = self.method_var.get()
        if choice == "Compare BE vs TR":
            return "compare"
        return choice  # "TR" or "BE"

    def _simulation_thread(self):
        """Executes the simulation off the main GUI thread."""
        try:
            method_choice = self._get_method_choice()

            if method_choice == "compare":
                circuit_tr, result_tr = self.run_simulation_core(
                    self.current_file_path, sensitivity=True, method='TR'
                )
                _, result_be = self.run_simulation_core(
                    self.current_file_path, sensitivity=True, method='BE'
                )
                self.root.after(0, lambda c=circuit_tr, r=result_tr, rc=result_be:
                    self._simulation_complete(c, r, rc))
            else:
                circuit, result = self.run_simulation_core(
                    self.current_file_path, sensitivity=True, method=method_choice
                )
                self.root.after(0, lambda c=circuit, r=result:
                    self._simulation_complete(c, r, None))

        except Exception as e:
            self.root.after(0, lambda e=e: self._simulation_error(e))

    def _simulation_error(self, error):
        """Restores the UI and displays exception tracebacks."""
        self.btn_run.config(state=tk.NORMAL, text="Run Simulation", bg="#90ee90")
        messagebox.showerror("Simulation Error", f"Simulation failed:\n\n{str(error)}")

    def _simulation_complete(self, circuit, result, result_compare):
        """Populates the UI after simulation completes."""
        self.circuit = circuit
        self.result = result
        self.result_compare = result_compare  # None or the BE result for comparison
        
        self.btn_run.config(state=tk.NORMAL, text="Run Simulation", bg="#90ee90")
        
        if not self.result:
            return

        # 1. Populate Node Listbox
        self.node_listbox.delete(0, tk.END)
        self.node_names_cache = []
        
        sorted_nodes = sorted(self.result.node_map.keys(), key=str)
        mna_prefixes = ('V', 'L', 'E', 'H', 'F')

        for node_name in sorted_nodes:
            self.node_names_cache.append(node_name)
            if str(node_name).upper().startswith(mna_prefixes):
                display_name = f"I({node_name})"
            else:
                display_name = f"V({node_name})"
            self.node_listbox.insert(tk.END, display_name)
            if not str(node_name).upper().startswith(mna_prefixes):
                self.node_listbox.selection_set(tk.END)

        # 2. Populate sensitivity parameter combobox
        available_params = set()
        for node in self.node_names_cache:
            available_params.update(self.result.get_sensitivity_parameters(node))
        self.sens_cb['values'] = ["None"] + sorted(list(available_params))
        self.sens_cb.current(0)

        # 3. Route to correct display
        if self.result.type == ".OP":
            self.update_data_tab_op()
            self.tabs.select(self.tab_data)
        else:
            self.update_data_tab_transient()
            self.update_plot()
            self.tabs.select(self.tab_plot)

    def on_selection_change(self, event):
        """Callback for listbox and combobox selections."""
        self.update_plot()

    def _format_sens(self, val):
        """Safely formats a sensitivity value that may be scalar or array."""
        if hasattr(val, '__len__'):
            val = float(val[0]) if len(val) == 1 else float(val[-1])
        return f"{val:+.6e}"

    def update_data_tab_op(self):
        """Displays DC Operating Point with sensitivities."""
        self.data_text.delete(1.0, tk.END)
        
        method_name = self.method_var.get()
        if method_name == "Compare BE vs TR":
            method_name = "TR (primary)"
        
        res_str = f"--- DC Operating Point [{method_name}] ---\n\n"
        
        for node in self.node_names_cache:
            val = self.result.get_voltage(node)
            unit = "A" if str(node).upper().startswith(('V', 'L')) else "V"
            res_str += f"{str(node):<10} | {val:+.6f} {unit}\n"
            
            calc_params = self.result.get_sensitivity_parameters(node)
            if calc_params:
                res_str += "  Sensitivities:\n"
                for comp in calc_params:
                    sens_val = self.result.get_sensitivity(node, comp)
                    res_str += f"    -> d({node})/d({comp}) = {self._format_sens(sens_val)}\n"
                res_str += "\n"
                
        self.data_text.insert(tk.END, res_str)

    def update_data_tab_transient(self):
        """Displays transient analysis data with BE vs TR companion model calculations."""
        if self.result.type != ".TRAN":
            return
            
        self.data_text.delete(1.0, tk.END)
        
        is_compare = self.result_compare is not None
        method_name = self.method_var.get()
        if method_name == "Compare BE vs TR":
            method_name = "TR"
        
        dt = self.result.dt
        t = self.result.sweep_axis
        num_steps = len(t)
        
        res_str = ""
        
        # ============================================================
        # SECTION 1: Companion Model Formulas & Computed Values
        # ============================================================
        res_str += "=" * 78 + "\n"
        res_str += "  COMPANION MODEL CALCULATIONS: Backward Euler vs Trapezoidal\n"
        res_str += "=" * 78 + "\n\n"
        res_str += f"  Time step dt = {dt:.4e} s\n"
        res_str += f"  Total steps  = {num_steps}\n"
        res_str += f"  Simulation   = {t[0]:.4e} s  to  {t[-1]:.4e} s\n\n"
        
        # Find capacitors and inductors
        caps = [c for c in self.circuit.components if c.type == 'C']
        inds = [c for c in self.circuit.components if c.type == 'L']
        
        if caps or inds:
            res_str += "-" * 78 + "\n"
            res_str += "  FORMULAS\n"
            res_str += "-" * 78 + "\n\n"
        
        for cap in caps:
            C = cap.value
            g_be = C / dt
            g_tr = 2.0 * C / dt
            
            res_str += f"  Capacitor {cap.name}:  C = {C:.4e} F\n\n"
            
            res_str += f"    Backward Euler (BE):\n"
            res_str += f"      G_eq = C/dt = {C:.4e} / {dt:.4e} = {g_be:.6e} S\n"
            res_str += f"      I_eq = G_eq * v_prev = {g_be:.4e} * v_prev\n"
            res_str += f"      i_new = G_eq * (v_new - v_prev)\n\n"
            
            res_str += f"    Trapezoidal (TR):\n"
            res_str += f"      G_eq = 2C/dt = 2*{C:.4e} / {dt:.4e} = {g_tr:.6e} S\n"
            res_str += f"      I_eq = i_prev + G_eq * v_prev\n"
            res_str += f"      i_new = G_eq * (v_new - v_prev) - i_prev\n\n"
            
            res_str += f"    Comparison:\n"
            res_str += f"      {'':30s} {'BE':>14s} {'TR':>14s} {'TR/BE':>10s}\n"
            res_str += f"      {'G_eq (Siemens)':<30s} {g_be:>14.6e} {g_tr:>14.6e} {g_tr/g_be:>10.2f}x\n"
            res_str += f"      {'History term':<30s} {'(C/dt)*v_prev':>14s} {'i_prev+(2C/dt)*v_prev':>14s}\n"
            res_str += f"      {'State stored':<30s} {'(not needed)':>14s} {'i_cap_prev':>14s}\n"
            res_str += f"      {'Accuracy order':<30s} {'O(dt)':>14s} {'O(dt²)':>14s}\n\n"
        
        for ind in inds:
            L = ind.value
            r_be = L / dt
            r_tr = 2.0 * L / dt
            
            res_str += f"  Inductor {ind.name}:  L = {L:.4e} H\n\n"
            
            res_str += f"    Backward Euler (BE):\n"
            res_str += f"      R_eq = L/dt = {L:.4e} / {dt:.4e} = {r_be:.6e} Ohm\n"
            res_str += f"      V_eq = R_eq * i_prev = {r_be:.4e} * i_prev\n"
            res_str += f"      v_L_new = R_eq * (i_new - i_prev)\n\n"
            
            res_str += f"    Trapezoidal (TR):\n"
            res_str += f"      R_eq = 2L/dt = 2*{L:.4e} / {dt:.4e} = {r_tr:.6e} Ohm\n"
            res_str += f"      V_eq = v_L_prev + R_eq * i_prev\n"
            res_str += f"      v_L_new = R_eq * (i_new - i_prev) - v_L_prev\n\n"
            
            res_str += f"    Comparison:\n"
            res_str += f"      {'':30s} {'BE':>14s} {'TR':>14s} {'TR/BE':>10s}\n"
            res_str += f"      {'R_eq (Ohms)':<30s} {r_be:>14.6e} {r_tr:>14.6e} {r_tr/r_be:>10.2f}x\n"
            res_str += f"      {'History term':<30s} {'(L/dt)*i_prev':>14s} {'v_prev+(2L/dt)*i_prev':>14s}\n"
            res_str += f"      {'State stored':<30s} {'(not needed)':>14s} {'v_L_prev':>14s}\n"
            res_str += f"      {'Accuracy order':<30s} {'O(dt)':>14s} {'O(dt²)':>14s}\n\n"
        
        # ============================================================
        # SECTION 2: Waveform Comparison at Key Time Points
        # ============================================================
        if is_compare:
            res_str += "-" * 78 + "\n"
            res_str += "  WAVEFORM COMPARISON AT KEY TIME POINTS\n"
            res_str += "-" * 78 + "\n\n"
            
            # Pick ~5 representative time indices
            indices = [0]
            for frac in [0.1, 0.25, 0.5, 0.75]:
                indices.append(int(frac * (num_steps - 1)))
            indices.append(num_steps - 1)
            indices = sorted(set(indices))
            
            # Show each voltage node
            for node in self.node_names_cache:
                if str(node).upper().startswith(('V', 'L', 'E')):
                    continue
                
                v_main = self.result.get_voltage(node)
                v_cmp = self.result_compare.get_voltage(node)
                
                res_str += f"  V({node}):\n"
                res_str += f"    {'Time':>12s} {'TR':>14s} {'BE':>14s} {'Diff (TR-BE)':>14s}\n"
                res_str += f"    {'-'*56}\n"
                
                for idx in indices:
                    vt = v_main[idx]
                    vb = v_cmp[idx]
                    diff = vt - vb
                    res_str += f"    {t[idx]:>12.4e} {vt:>+14.6f} {vb:>+14.6f} {diff:>+14.6e}\n"
                
                max_diff = max(abs(v_main - v_cmp))
                res_str += f"    {'Max |diff|':>12s} {'':>14s} {'':>14s} {max_diff:>14.6e}\n\n"
        
        # ============================================================
        # SECTION 3: Sensitivity Comparison
        # ============================================================
        has_sensitivity = False
        for node in self.node_names_cache:
            if self.result.get_sensitivity_parameters(node):
                has_sensitivity = True
                break
        
        if has_sensitivity:
            res_str += "-" * 78 + "\n"
            res_str += "  INTEGRATED SENSITIVITY COMPARISON\n"
            res_str += "-" * 78 + "\n\n"
            
            for node in self.node_names_cache:
                calc_params = self.result.get_sensitivity_parameters(node)
                if not calc_params:
                    continue
                    
                unit = "A" if str(node).upper().startswith(('V', 'L')) else "V"
                v_final = self.result.get_voltage(node)
                if hasattr(v_final, '__len__'):
                    v_final = v_final[-1]
                res_str += f"  Node {node}  (final value: {v_final:+.6f} {unit})\n\n"
                
                if is_compare:
                    res_str += f"    {'Parameter':<15s} {'TR':>18s} {'BE':>18s} {'Diff':>14s} {'%Diff':>10s}\n"
                    res_str += f"    {'-'*75}\n"
                else:
                    res_str += f"    {'Parameter':<15s} {method_name+' Sensitivity':>18s}\n"
                    res_str += f"    {'-'*35}\n"
                
                for comp in calc_params:
                    s_main = self.result.get_sensitivity(node, comp, output_format="integrated")
                    if hasattr(s_main, '__len__'):
                        s_main = float(s_main[0]) if len(s_main) == 1 else float(s_main[-1])
                    
                    if is_compare:
                        try:
                            s_cmp = self.result_compare.get_sensitivity(node, comp, output_format="integrated")
                            if hasattr(s_cmp, '__len__'):
                                s_cmp = float(s_cmp[0]) if len(s_cmp) == 1 else float(s_cmp[-1])
                        except Exception:
                            s_cmp = float('nan')
                        
                        diff = s_main - s_cmp
                        pct = (diff / s_cmp * 100) if abs(s_cmp) > 1e-30 else float('nan')
                        res_str += f"    {comp:<15s} {s_main:>+18.6e} {s_cmp:>+18.6e} {diff:>+14.4e} {pct:>+10.1f}%\n"
                    else:
                        res_str += f"    {comp:<15s} {s_main:>+18.6e}\n"
                
                res_str += "\n"
        
        # ============================================================
        # SECTION 4: Theory Summary
        # ============================================================
        res_str += "=" * 78 + "\n"
        res_str += "  THEORY: WHY TR IS MORE ACCURATE THAN BE\n"
        res_str += "=" * 78 + "\n\n"
        res_str += "  Backward Euler approximates the derivative by looking backward:\n"
        res_str += "    dy/dt ≈ (y[n] - y[n-1]) / dt\n\n"
        res_str += "  Trapezoidal uses the average of both endpoints:\n"
        res_str += "    dy/dt ≈ (y[n] - y[n-1]) / dt\n"
        res_str += "    y     ≈ (y[n] + y[n-1]) / 2\n\n"
        res_str += "  s-to-z mappings (from MIT 6.003 Lecture 7):\n"
        res_str += "    BE:  s → (z-1)/(T*z)        — 1st order, unconditionally stable\n"
        res_str += "    TR:  s → 2(z-1)/(T*(z+1))   — 2nd order, unconditionally stable\n\n"
        res_str += "  Convergence when halving dt:\n"
        res_str += "    BE error halves    (ratio ≈ 2)  — O(dt)\n"
        res_str += "    TR error quarters  (ratio ≈ 4)  — O(dt²)\n\n"
        res_str += "  Stability regions:\n"
        res_str += "    BE: entire left half-plane → inside circle at z=1/2, radius 1/2\n"
        res_str += "    TR: entire left half-plane → inside unit circle (exact mapping)\n"
        res_str += "        jω axis → unit circle (can cause ringing on sharp edges)\n"
        
        self.data_text.insert(tk.END, res_str)

    def update_plot(self):
        """Dynamically draws Bode, Transient, DC Sweep, and Sensitivity plots."""
        if not self.result: 
            return

        self.fig.clf() 
        
        selected_indices = self.node_listbox.curselection()
        if not selected_indices:
            self.canvas.draw()
            return

        x_axis = self.result.sweep_axis
        target_comp = self.sens_cb.get()
        is_compare = self.result_compare is not None
        
        method_name = self.method_var.get()
        if method_name == "Compare BE vs TR":
            method_name = "TR"

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
                
                v = self.result.get_voltage(node_name)
                mag_db = 20 * np.log10(np.where(np.abs(v) == 0, 1e-12, np.abs(v)))
                phase = np.angle(v, deg=True)
                
                ax1.semilogx(x_axis, mag_db, label=label, lw=2)
                ax2.semilogx(x_axis, phase, label=label, lw=2)
                
                if show_sens and target_comp in self.result.get_sensitivity_parameters(node_name):
                    raw_sens = self.result.get_sensitivity(node_name, target_comp)
                    ax3.semilogx(x_axis, np.abs(raw_sens), label=f"|d({label})/d{target_comp}|", lw=2, linestyle='--')

            ax1.set_ylabel("Magnitude (dB)")
            ax1.set_title("AC Analysis: Bode Plot")
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
            
            x_label = "Time (s)" if self.result.type == ".TRAN" else "Sweep Voltage (V)"
            
            for listbox_idx in selected_indices:
                node_name = self.node_names_cache[listbox_idx]
                label = self.node_listbox.get(listbox_idx)
                
                v = self.result.get_voltage(node_name)
                ax1.plot(x_axis, v, label=f"{label} [{method_name}]", lw=2)
                
                # Overlay comparison waveform if in compare mode
                if is_compare and self.result.type == ".TRAN":
                    v_cmp = self.result_compare.get_voltage(node_name)
                    ax1.plot(x_axis, v_cmp, label=f"{label} [BE]", lw=1.5, linestyle='--', alpha=0.7)
                
                if show_sens and target_comp in self.result.get_sensitivity_parameters(node_name):
                    sens_fmt = "series" if self.result.type == ".TRAN" else None
                    
                    if sens_fmt:
                        s_main = self.result.get_sensitivity(node_name, target_comp, output_format=sens_fmt)

                        # Sensitivity series is now always aligned with sweep_axis
                        ax2.plot(x_axis, s_main, label=f"d({label})/d({target_comp}) [{method_name}]", lw=2)

                        if is_compare:
                            try:
                                s_cmp = self.result_compare.get_sensitivity(node_name, target_comp, output_format=sens_fmt)
                                ax2.plot(x_axis, s_cmp, label=f"d({label})/d({target_comp}) [BE]", lw=1.5, linestyle='--', alpha=0.7)
                            except Exception:
                                pass
                    else:
                        s_main = self.result.get_sensitivity(node_name, target_comp)
                        ax2.plot(x_axis, s_main, label=f"d({label})/d({target_comp})", lw=2, linestyle='--')

            ax1.set_ylabel("Voltage (V) / Current (A)")
            ax1.grid(True, ls="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=8)
            
            if show_sens:
                ax2.set_ylabel(f"Sensitivity w.r.t {target_comp}")
                ax2.set_xlabel(x_label)
                ax2.grid(True, ls="--", alpha=0.5)
                ax2.legend(loc="upper right", fontsize=8)
            else:
                ax1.set_xlabel(x_label)

        self.fig.tight_layout()
        self.canvas.draw()