"""
Graphical User Interface Module.

This module provides a Tkinter-based front-end for the SPICE simulator. It allows
users to load netlists, run simulations asynchronously, and visualize the output
using embedded Matplotlib figures. It also includes an interactive SDWC Yield Dashboard.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import sys
import os
from scipy.stats import norm

# Adjust this import to match where you saved your yield functions!
from applications.yield_analysis import perform_sdwc_yield_analysis, _get_nominal_values


class ThreadSafeConsole:
    """Redirects sys.stdout to a Tkinter Text widget safely across threads."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.after(0, self._insert_text, text)

    def _insert_text(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class CircuitSimulatorGUI:
    """The main GUI application for the Python SPICE Simulator."""

    def __init__(self, root, simulation_callback):
        self.root = root
        self.root.title("Python SPICE Simulator")
        self.root.geometry("1400x900") # Made slightly wider for the dashboard
        
        self.run_simulation_core = simulation_callback

        self.circuit = None 
        self.result = None
        self.live_yield_data = None # Holds the data vault for the live sliders

        # --- Top Control Panel ---
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_load = tk.Button(control_frame, text="Load Netlist", command=self.load_file, width=15)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.lbl_file = tk.Label(control_frame, text="No file selected", fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        # Yield Button
        self.btn_yield = tk.Button(control_frame, text="Run Yield Analysis", command=self.open_yield_dialog, width=20, bg="#ffb6c1", state=tk.DISABLED)
        self.btn_yield.pack(side=tk.RIGHT, padx=5)

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

        # --- Tab 3: Yield Analysis Console ---
        self.tab_yield = tk.Frame(self.tabs)
        self.tabs.add(self.tab_yield, text="Yield Console")
        self.yield_text = tk.Text(self.tab_yield, font=("Courier", 10), bg="#1e1e1e", fg="#00ff00")
        self.yield_text.pack(fill=tk.BOTH, expand=True)

        # === INSIDE __init__ (Replace the Yield Dashboard Frame setup) ===
        # --- Tab 4: LIVE YIELD DASHBOARD ---
        self.tab_dashboard = tk.Frame(self.tabs)
        self.tabs.add(self.tab_dashboard, text="Live Yield Dashboard")
        
        dash_ctrl = tk.Frame(self.tab_dashboard, width=250, bg="#f5f5f5")
        dash_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        dash_plots = tk.Frame(self.tab_dashboard)
        dash_plots.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(dash_ctrl, text="Interactive Parameters", font=("Arial", 12, "bold"), bg="#f5f5f5").pack(pady=10)

        # NEW: Time-Step Selector Dropdown
        tk.Label(dash_ctrl, text="Evaluation Point:", font=("Arial", 10, "bold"), bg="#f5f5f5").pack(pady=(5, 0))
        self.cb_eval_step = ttk.Combobox(dash_ctrl, state="readonly", font=("Arial", 9))
        self.cb_eval_step.pack(fill=tk.X, padx=10, pady=5)
        self.cb_eval_step.bind("<<ComboboxSelected>>", self._on_slider_change)

        # Instant-Feedback Sliders
        self.slider_tol = tk.Scale(dash_ctrl, from_=1.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, label="Factory Tol (±%)", bg="#f5f5f5", command=self._on_slider_change)
        self.slider_tol.set(5.0)
        self.slider_tol.pack(fill=tk.X, padx=10, pady=5)

        self.slider_spec = tk.Scale(dash_ctrl, from_=1.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, label="Spec Window (±%)", bg="#f5f5f5", command=self._on_slider_change)
        self.slider_spec.set(5.0)
        self.slider_spec.pack(fill=tk.X, padx=10, pady=5)

        self.slider_sigma = tk.Scale(dash_ctrl, from_=1.0, to=6.0, resolution=0.1, orient=tk.HORIZONTAL, label="Manufacturing Sigma", bg="#f5f5f5", command=self._on_slider_change)
        self.slider_sigma.set(6.0)
        self.slider_sigma.pack(fill=tk.X, padx=10, pady=5)

        self.slider_tgt_sigma = tk.Scale(dash_ctrl, from_=1.0, to=6.0, resolution=0.1, orient=tk.HORIZONTAL, label="Target Yield Sigma", bg="#f5f5f5", command=self._on_slider_change)
        self.slider_tgt_sigma.set(6.0)
        self.slider_tgt_sigma.pack(fill=tk.X, padx=10, pady=5)

        # Live Stats Display
        self.lbl_yield_stat = tk.Label(dash_ctrl, text="Yield: -- %", font=("Arial", 14, "bold"), fg="green", bg="#f5f5f5")
        self.lbl_yield_stat.pack(pady=(15, 0))
        self.lbl_dpmo_stat = tk.Label(dash_ctrl, text="DPMO: --", font=("Arial", 12), bg="#f5f5f5")
        self.lbl_dpmo_stat.pack(pady=(0, 10))
        
        self.lbl_status = tk.Label(dash_ctrl, text="STATUS: --", font=("Arial", 12, "bold"), bg="#f5f5f5")
        self.lbl_status.pack()

        # Dashboard Matplotlib Setup
        self.dash_fig = Figure(figsize=(10, 8), dpi=100)
        self.dash_canvas = FigureCanvasTkAgg(self.dash_fig, master=dash_plots)
        self.dash_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax_env = self.dash_fig.add_subplot(221)
        self.ax_pdf = self.dash_fig.add_subplot(222)
        self.ax_pareto = self.dash_fig.add_subplot(212)
        self.dash_fig.tight_layout(pad=3.0)

        self.current_file_path = None
        self.node_names_cache = []
        self.step_mapping = {} # Resolves Combobox string back to step index

    # =========================================================================
    # SIMULATION ROUTINES
    # =========================================================================
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.current_file_path = file_path
            self.lbl_file.config(text=file_path.split("/")[-1], fg="black")
            self.btn_run.config(state=tk.NORMAL)
            self.btn_yield.config(state=tk.DISABLED)
            with open(file_path, "r") as f:
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, f.read())

    def run_simulation(self):
        if self.current_file_path:
            with open(self.current_file_path, "w") as f:
                f.write(self.text_area.get(1.0, tk.END))
        else:
            messagebox.showerror("Error", "No file loaded.")
            return

        self.btn_run.config(state=tk.DISABLED, text="Simulating...", bg="#cccccc")
        self.btn_yield.config(state=tk.DISABLED)
        threading.Thread(target=self._simulation_thread, daemon=True).start()

    def _simulation_thread(self):
        try:
            self.circuit, self.result = self.run_simulation_core(
                self.current_file_path, 
                sensitivity=True
            )
            self.root.after(0, self._simulation_complete)
        except Exception as e:
            self.root.after(0, lambda e=e: self._simulation_error(e))

    def _simulation_error(self, error):
        self.btn_run.config(state=tk.NORMAL, text="Run Simulation", bg="#90ee90")
        messagebox.showerror("Simulation Error", f"Simulation failed:\n\n{str(error)}")

    def _simulation_complete(self):
        self.btn_run.config(state=tk.NORMAL, text="Run Simulation", bg="#90ee90")
        if not self.result: return

        if self.result.type in [".TRAN", ".DC"]:
            self.btn_yield.config(state=tk.NORMAL)

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

        available_params = set()
        for node in self.node_names_cache:
            available_params.update(self.result.get_sensitivity_parameters(node))
            
        self.sens_cb['values'] = ["None"] + sorted(list(available_params))
        self.sens_cb.current(0)

        if self.result.type == ".OP":
            self.update_data_tab_op()
            self.tabs.select(self.tab_data)
        else:
            self.update_plot()
            self.tabs.select(self.tab_plot)

    # =========================================================================
    # YIELD ANALYSIS GUI COMPONENTS
    # =========================================================================
    def open_yield_dialog(self):
        if not self.result: return

        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Yield Analysis")
        dialog.geometry("320x150") # Shortened since sliders handle the rest
        dialog.resizable(False, False)

        tk.Label(dialog, text="Target Evaluation Node:").pack(pady=(10, 0))
        node_cb = ttk.Combobox(dialog, values=self.node_names_cache, state="readonly")
        node_cb.current(0)
        node_cb.pack()

        def launch_yield():
            config = {
                "node": node_cb.get(),
                "tol": self.slider_tol.get() / 100.0,
                "spec": self.slider_spec.get() / 100.0,
                "mfg_sigma": self.slider_sigma.get(),
                "tgt_sigma": self.slider_tgt_sigma.get()
            }
            dialog.destroy()
            self.tabs.select(self.tab_yield)
            
            self.btn_run.config(state=tk.DISABLED)
            self.btn_yield.config(state=tk.DISABLED, text="Scouting Yield...")
            self.yield_text.insert(tk.END, "\n======================================================\n")
            self.yield_text.insert(tk.END, f"INITIALIZING SDWC YIELD ENGINE FOR {config['node']}...\n")
            self.yield_text.insert(tk.END, "======================================================\n")
            
            threading.Thread(target=self._yield_thread, args=(config,), daemon=True).start()

        tk.Button(dialog, text="Start SDWC Engine", command=launch_yield, bg="#ffb6c1", width=20).pack(pady=20)

    def _yield_thread(self, config):
        old_stdout = sys.stdout
        sys.stdout = ThreadSafeConsole(self.yield_text)

        try:
            calc_params = []
            for comp in self.circuit.components:
                calc_params.extend(comp.differentiable_params)
            
            netlist_name = os.path.basename(self.current_file_path).split('.')[0]

            # Capture the data vault returned by the engine
            vault = perform_sdwc_yield_analysis(
                circuit=self.circuit, 
                result=self.result, 
                target_node=config["node"], 
                calculated_params=calc_params, 
                netlist_name=netlist_name, 
                folder_map={},  
                factory_tolerance=config["tol"], 
                sigma_level=config["mfg_sigma"],
                out_spec=config["spec"], 
                out_sigma_level=config["tgt_sigma"], 
                k_params=4 
            )
            
            # Store configuration so slider math knows what we are analyzing
            vault['target_node'] = config["node"]
            vault['orig_tol'] = config["tol"]
            vault['orig_sigma'] = config["mfg_sigma"]
            
            # Push the data to the main thread safely
            self.root.after(0, self._yield_complete, vault)
            
        except Exception as e:
            print(f"\n[ERROR] Yield Engine Failed: {str(e)}")
            self.root.after(0, self._restore_yield_buttons)
        finally:
            sys.stdout = old_stdout

    def _restore_yield_buttons(self):
        self.btn_run.config(state=tk.NORMAL)
        self.btn_yield.config(state=tk.NORMAL, text="Run Yield Analysis")

    # === INSIDE _yield_complete (Replaces the old _yield_complete) ===
    def _yield_complete(self, vault):
        self._restore_yield_buttons()
        print("\n[SUCCESS] Yield Analysis Complete. Switching to Live Dashboard!")
        self.live_yield_data = vault
        
        # Populate the new Combobox with ALL identified evaluation points
        eval_steps = vault["evaluation_steps"]
        t_axis = self.result.sweep_axis * 1e9
        
        cb_values = []
        self.step_mapping.clear()
        default_str = ""
        
        for step, labels in eval_steps.items():
            label_str = "+".join([lbl.split('_')[0] for lbl in labels])
            display_str = f"{label_str} (t={t_axis[step]:.2f}ns)"
            cb_values.append(display_str)
            self.step_mapping[display_str] = step
            
            if step == vault["default_step"]:
                default_str = display_str
                
        self.cb_eval_step['values'] = cb_values
        if default_str:
            self.cb_eval_step.set(default_str)
        else:
            self.cb_eval_step.current(0)
            
        # Switch tabs and trigger the first draw
        self.tabs.select(self.tab_dashboard)
        self._update_live_dashboard()

    # =========================================================================
    # LIVE DASHBOARD ENGINE
    # =========================================================================
    def _on_slider_change(self, event=None):
        """Callback triggered by Tkinter when a user drags a slider."""
        self._update_live_dashboard()

    # === INSIDE _update_live_dashboard (Replaces the old _update_live_dashboard) ===
    def _update_live_dashboard(self):
        """Fast O(1) mathematical recalculation and Matplotlib redraw loop."""
        if not self.live_yield_data: return

        selected_str = self.cb_eval_step.get()
        if not selected_str: return
        
        data = self.live_yield_data
        step_idx = self.step_mapping[selected_str] # <--- DYNAMIC STEP INDEX!
        target_node = data['target_node']

        # 1. Read Live Sliders
        tol = self.slider_tol.get() / 100.0
        spec = self.slider_spec.get() / 100.0
        mfg_sigma = self.slider_sigma.get()
        tgt_sigma = self.slider_tgt_sigma.get()
        
        v_nom = self.result.VI[step_idx][self.result.node_map[target_node]]
        spec_min = v_nom * (1.0 - spec)
        spec_max = v_nom * (1.0 + spec)

        # 2. Fast Adjoint Recalculation (Linear Math)
        p_noms = _get_nominal_values(self.circuit, [d['param'] for d in data['full_ranking']])
        o_idx = self.result.sensitivities.output_index[target_node]
        
        step_ranking = []
        for d, p_nom in zip(data['full_ranking'], p_noms):
            p_idx = self.result.sensitivities.param_index[d['param']]
            
            # Fetch the RAW Adjoint derivative specifically for THIS time step!
            raw_sens = self.result.sensitivities.data[p_idx, o_idx, step_idx]
            
            p_sigma = (p_nom * tol) / mfg_sigma
            dv_expected = np.abs(raw_sens * p_sigma)
            
            step_ranking.append({'param': d['param'], 'dv_expected': dv_expected})

        step_ranking.sort(key=lambda x: x['dv_expected'], reverse=True)
        
        # New Circuit Variance
        sigma_out = np.sqrt(np.sum([d['dv_expected']**2 for d in step_ranking]))
        
        prob_passing = norm.cdf(spec_max, loc=v_nom, scale=sigma_out) - norm.cdf(spec_min, loc=v_nom, scale=sigma_out)
        yield_pct = prob_passing * 100.0
        dpmo = (1.0 - prob_passing) * 1_000_000

        max_allow_sigma = (v_nom * spec) / tgt_sigma
        if sigma_out <= max_allow_sigma:
            status_text, status_color = "STATUS: PASS", "green"
        else:
            status_text, status_color = "STATUS: FAIL", "red"

        # Update Text
        yield_color = "green" if yield_pct > 99.99 else "darkorange" if yield_pct > 95.0 else "red"
        self.lbl_yield_stat.config(text=f"Yield: {yield_pct:.4f}%", fg=yield_color)
        self.lbl_dpmo_stat.config(text=f"DPMO: {dpmo:,.0f}")
        self.lbl_status.config(text=status_text, fg=status_color)

        # 3. REDRAW PDF
        self.ax_pdf.clear()
        x_axis = np.linspace(v_nom - 6*sigma_out, v_nom + 6*sigma_out, 500)
        pdf_values = norm.pdf(x_axis, v_nom, sigma_out)
        
        self.ax_pdf.plot(x_axis, pdf_values, color='black', lw=2)
        pass_region = (x_axis >= spec_min) & (x_axis <= spec_max)
        self.ax_pdf.fill_between(x_axis, pdf_values, where=pass_region, color='green', alpha=0.2)
        
        fail_low, fail_high = (x_axis < spec_min), (x_axis > spec_max)
        self.ax_pdf.fill_between(x_axis, pdf_values, where=fail_low, color='red', alpha=0.4)
        self.ax_pdf.fill_between(x_axis, pdf_values, where=fail_high, color='red', alpha=0.4)
        
        self.ax_pdf.axvline(spec_min, color='red', linestyle='--', lw=2)
        self.ax_pdf.axvline(spec_max, color='red', linestyle='--', lw=2)
        self.ax_pdf.set_title(f"Gaussian Yield Density at t={self.result.sweep_axis[step_idx]*1e9:.2f}ns", fontsize=10, fontweight='bold')
        self.ax_pdf.grid(True, alpha=0.3)

        # 4. REDRAW ENVELOPE
        self.ax_env.clear()
        t_axis = self.result.sweep_axis * 1e9
        n_idx = self.result.node_map[target_node]
        v_nominal_wave = self.result.VI[:, n_idx]
        
        self.ax_env.fill_between(t_axis, data['v_min'], data['v_max'], color='red', alpha=0.2)
        self.ax_env.plot(t_axis, v_nominal_wave, color='black', linewidth=2)
        
        dynamic_spec_min = v_nominal_wave * (1.0 - spec)
        dynamic_spec_max = v_nominal_wave * (1.0 + spec)
        self.ax_env.plot(t_axis, dynamic_spec_max, color='green', linestyle='--', lw=1.5)
        self.ax_env.plot(t_axis, dynamic_spec_min, color='green', linestyle='--', lw=1.5)
        
        self.ax_env.axvline(t_axis[step_idx], color='blue', linestyle='-', lw=2) # Made line solid for visibility
        self.ax_env.set_title(f"Envelope at V({target_node})", fontsize=10, fontweight='bold')
        self.ax_env.grid(True, alpha=0.3)

        # 5. REDRAW PARETO (Dual-Bar)
        self.ax_pareto.clear()
        clean_data = [d for d in step_ranking if d['dv_expected'] > 1e-9][:10]
        clean_data.reverse()

        params = [d['param'] for d in clean_data]
        adj_vals_mv = [d['dv_expected'] * 1000 for d in clean_data] 
        
        orig_tol = data.get('orig_tol', 0.05)
        orig_sigma = data.get('orig_sigma', 6.0)
        wb_scale = (tol / orig_tol) * (orig_sigma / mfg_sigma)

        wb_vals_mv = []
        # CRITICAL: Dynamically grab the exact Woodbury Vault for the selected dropdown step!
        wb_deltas = data.get('all_woodbury_deltas', {}).get(step_idx, {}) 
        
        for p in params:
            if p in wb_deltas:
                wb_vals_mv.append(np.abs(wb_deltas[p]) * wb_scale * 1000)
            else:
                wb_vals_mv.append(0.0)

        y = np.arange(len(params))
        height = 0.4 
        rects1 = self.ax_pareto.barh(y + height/2, adj_vals_mv, height, label='Adjoint Prediction', color='lightblue', edgecolor='black')
        rects2 = self.ax_pareto.barh(y - height/2, wb_vals_mv, height, label='Woodbury Truth', color='coral', edgecolor='black')
        
        self.ax_pareto.set_yticks(y)
        self.ax_pareto.set_yticklabels(params)
        self.ax_pareto.set_title(f"Live 1σ Sensitivity (mV) at {selected_str}", fontsize=10, fontweight='bold')
        self.ax_pareto.grid(axis='x', linestyle='--', alpha=0.6)
        self.ax_pareto.legend(loc='lower right', framealpha=0.9)

        max_val = max(max(adj_vals_mv) if adj_vals_mv else 0.0, max(wb_vals_mv) if wb_vals_mv else 0.0)
        max_val = max(max_val, 1.0) # Prevent 0 bounds
            
        for rect in rects1:
            width = rect.get_width()
            self.ax_pareto.text(width + (max_val * 0.01), rect.get_y() + rect.get_height()/2, 
                    f'{width:.2f}', va='center', fontweight='bold', color='black', fontsize=8)

        for rect in rects2:
            width = rect.get_width()
            if width > 0.0:
                self.ax_pareto.text(width + (max_val * 0.01), rect.get_y() + rect.get_height()/2, 
                        f'{width:.2f}', va='center', fontweight='bold', color='darkred', fontsize=8)

        self.ax_pareto.set_xlim(0, max_val * 1.15)
        self.dash_canvas.draw_idle()


    # =========================================================================
    # STANDARD PLOT UPDATE ROUTINES (Unchanged)
    # =========================================================================
    def on_selection_change(self, event):
        self.update_plot()

    def update_data_tab_op(self):
        self.data_text.delete(1.0, tk.END)
        res_str = "--- DC Operating Point ---\n"
        for node in self.node_names_cache:
            val = self.result.get_voltage(node)
            unit = "A" if str(node).upper().startswith(('V', 'L')) else "V"
            res_str += f"{str(node):<10} | {val:+.6f} {unit}\n"
            
            calc_params = self.result.get_sensitivity_parameters(node)
            if calc_params:
                res_str += "  Sensitivities:\n"
                for comp in calc_params:
                    sens_val = self.result.get_sensitivity(node, comp)
                    res_str += f"    -> d({node})/d({comp}) = {sens_val:+.6e}\n"
                res_str += "\n"
        self.data_text.insert(tk.END, res_str)

    def update_plot(self):
        if not self.result: return
        self.fig.clf() 
        
        selected_indices = self.node_listbox.curselection()
        if not selected_indices:
            self.canvas.draw()
            return

        x_axis = self.result.sweep_axis
        target_comp = self.sens_cb.get()

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
                
                v = self.result.get_voltage(node_name)
                ax1.plot(x_axis, v, label=label, lw=2)
                
                if show_sens and target_comp in self.result.get_sensitivity_parameters(node_name):
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
