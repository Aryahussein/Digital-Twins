"""
Simulation Plotting and Output Utility Module.

This module provides visualization and console output tools for the EDA simulator. 
It takes `SimulationResult` objects and generates industry-standard Bode plots, 
transient waveforms, DC sweeps, sensitivity curves, and formatted operating 
point summaries.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come before importing pyplot for headless environments
import matplotlib.pyplot as plt
from applications.large_change_sensitivity import build_xi, compute_large_change

# ==========================================
# AC PLOTTING (FREQUENCY DOMAIN)
# ==========================================

def make_bode_plot(result, output_nodes, folder="./figures/ac", name="bodeplot"):
    """Plots standard Magnitude (dB) and Phase (degrees) vs Frequency."""
    if not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    
    print(f"Plotting Bode plot for output node(s) {output_nodes}...")
    frequencies = result.sweep_axis
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    for node in output_nodes:
        V_out = result.get_voltage(node) 
        
        mag = np.abs(V_out)
        mag = np.where(mag == 0, 1e-12, mag)
        mag_db = 20 * np.log10(mag)
        phase = np.angle(V_out, deg=True)
        
        ax1.semilogx(frequencies, mag_db, lw=2, label=f"Node {node}")
        ax2.semilogx(frequencies, phase, lw=2, label=f"Node {node}")
    
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("AC Analysis: Bode Plot")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend(loc="best")

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (Degrees)")
    ax2.set_yticks(np.arange(-180, 181, 45))
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend(loc="best")
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_ac_sensitivity(result, output_node, target_component, folder="./figures/ac", name="ac_sensitivity"):
    """Plots Output Magnitude alongside the Adjoint Sensitivity Magnitude."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]
    
    print(f"Plotting AC sensitivity for output node {output_node} w.r.t {target_component}...")
    
    frequencies = result.sweep_axis
    V_out = result.get_voltage(output_node)
    
    mag = np.abs(V_out)
    mag = np.where(mag == 0, 1e-12, mag)
    mag_db = 20 * np.log10(mag)

    raw_sens = result.get_sensitivity(output_node, target_component)
    sens_mags = np.abs(raw_sens)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    ax1.semilogx(frequencies, mag_db, lw=2, color='blue')
    ax1.set_ylabel("Output Mag (dB)")
    ax1.set_title(f"AC Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    ax2.semilogx(frequencies, sens_mags, lw=2, color='red')
    ax2.set_ylabel(f"| dV_{output_node} / d{target_component} |")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

# ==========================================
# TRANSIENT PLOTTING (TIME DOMAIN)
# ==========================================

def plot_transient(result, output_nodes=None, folder="./figures/tran", name="transient"):
    """Plots standard Voltage vs Time waveforms."""
    print(f"Plotting transient response...")
    
    time = result.sweep_axis
    fig, ax = plt.subplots(figsize=(6, 4))

    if output_nodes is None:
        output_nodes = [k for k in result.node_map.keys() if not str(k).upper().startswith(('V', 'L', 'E', 'H', 'F'))]
    elif not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    
    for node in output_nodes:
        V_out = result.get_voltage(node) 
        ax.plot(time, V_out, lw=2, label=f"Node {node}")

    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Transient Response")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend(loc="best")
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_transient_sensitivity(result, output_node, target_component, folder="./figures/tran", name="tran_sensitivity"):
    """Plots Transient Voltage alongside the Transient Sensitivity series over time."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0] 
        
    print(f"Plotting transient sensitivity for output node {output_node} w.r.t {target_component}...")
    
    time = result.sweep_axis
    V_out = result.get_voltage(output_node)
    sens_array = result.get_sensitivity(output_node, target_component)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    ax1.plot(time, V_out, lw=2, color='blue')
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Transient Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)

    ax2.plot(time, sens_array, lw=2, color='red')
    ax2.set_ylabel(f"dV_{output_node} / d{target_component}")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, ls="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

# ==========================================
# DC SWEEP PLOTTING
# ==========================================

def plot_dc_sweep(result, output_nodes=None, folder="./figures/dc", name="dc_sweep"):
    """Plots Voltage/Current vs DC Sweep Voltage."""
    print(f"Plotting DC sweep response...")
    
    sweep_v = result.sweep_axis
    fig, ax = plt.subplots(figsize=(6, 4))

    if output_nodes is None:
        output_nodes = [k for k in result.node_map.keys() if not str(k).upper().startswith(('V', 'L', 'E', 'H', 'F'))]
    elif not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    
    for node in output_nodes:
        V_out = result.get_voltage(node) 
        ax.plot(sweep_v, V_out, lw=2, label=f"Node {node}")

    ax.set_ylabel("Output Voltage (V) / Current (A)")
    ax.set_xlabel("Sweep Source (V/A)")
    ax.set_title("DC Sweep Response")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend(loc="best")
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_dc_sensitivity(result, output_node, target_component, folder="./figures/dc", name="dc_sensitivity"):
    """Plots DC Sweep Voltage alongside the DC Sensitivity series."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0] 
        
    print(f"Plotting DC sensitivity for output node {output_node} w.r.t {target_component}...")
    
    sweep_v = result.sweep_axis
    V_out = result.get_voltage(output_node)
    sens_array = result.get_sensitivity(output_node, target_component)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    ax1.plot(sweep_v, V_out, lw=2, color='blue')
    ax1.set_ylabel("Output (V/A)")
    ax1.set_title(f"DC Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)

    ax2.plot(sweep_v, sens_array, lw=2, color='red')
    ax2.set_ylabel(f"dV_{output_node} / d{target_component}")
    ax2.set_xlabel("Sweep Source (V/A)")
    ax2.grid(True, ls="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

# ==========================================
# CONSOLE OUTPUT (.OP)
# ==========================================

def print_solution(result):
    """Prints the operating point solution vector nicely to the console."""
    is_ac = result.type == ".AC"
    header = f"AC @ {result.sweep_axis[0]:.2e} Hz" if is_ac else "DC / OP"
    print(f"\n--- Simulation Results ({header}) ---")
    
    nodes = []
    branches = []
    
    mna_prefixes = ('V', 'L', 'E', 'H', 'F')
    
    for key, idx in result.node_map.items():
        if str(key).upper().startswith(mna_prefixes):
            branches.append((key, idx))
        else:
            nodes.append((key, idx))

    # V = result.VI
    V = result.VI.flatten() if result.VI.ndim > 1 else result.VI

    print("Node Voltages:")
    for node, idx in sorted(nodes, key=lambda x: str(x[0])):
        val = V[idx]
        if not is_ac:
            print(f"  Node {node:5}: {val.real:10.6f} V")
        else:
            mag = np.abs(val)
            phase = np.degrees(np.angle(val))
            print(f"  Node {node:5}: {mag:10.6f} V ∠ {phase:7.2f}°")

    if branches:
        print("\nBranch Currents:")
        for name, idx in sorted(branches, key=lambda x: str(x[0])):
            val = V[idx]
            if not is_ac:
                print(f"  {name:7}: {val.real:10.6f} A")
            else:
                mag = np.abs(val)
                phase = np.degrees(np.angle(val))
                print(f"  {name:7}: {mag:10.6f} A ∠ {phase:7.2f}°")


def plot_fault_comparison(circuit, result, output_node, threshold_results, delta_F,
                           folder="./figures/fault", name="fault_comparison"):
    """
    Plot nominal, shorted, and opened circuit responses on one graph.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    
    tensor = result.sensitivities
    time = tensor.sweep_axis
    num_steps = len(time)
    n = circuit.total_dim
    out_idx = circuit.get_idx(output_node)
    
    # --- Get nominal V(out) at every time step ---
    v_nominal = np.zeros(num_steps)
    for t_idx in range(num_steps):
        v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI
        v_nominal[t_idx] = np.real(v[out_idx])
    
    # --- Compute faulted V(out) for the top short ---
    shorts = threshold_results.get("shorts", [])
    v_short = None
    short_label = None
    short_best_time = None
    
    if shorts and shorts[0]['threshold_R'] is not None:
        s = shorts[0]
        short_label = f"Short {s['location']} (R={s['threshold_R']:.0f}Ω)"
        short_best_time = s['best_time']
        
        # Parse node indices
        parts = s['location'].split("<->")
        n1_name, n2_name = parts[0], parts[1]
        idx_k = circuit.node_map.get(n1_name)
        idx_l = circuit.node_map.get(n2_name)
        if idx_k is None:
            try: idx_k = circuit.node_map.get(int(n1_name))
            except ValueError: pass
        if idx_l is None:
            try: idx_l = circuit.node_map.get(int(n2_name))
            except ValueError: pass
        
        if idx_k is not None or idx_l is not None:
            xi_kl = build_xi(n, idx_k, idx_l)
            R_short = s['threshold_R']
            
            v_short = np.zeros(num_steps)
            for t_idx in range(num_steps):
                lu = result.list_of_lus[t_idx]
                v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI
                delta_v = compute_large_change(lu, xi_kl, v, R_short, out_idx)
                v_short[t_idx] = np.real(v[out_idx] + delta_v)
    
    # --- Compute faulted V(out) for the top open ---
    opens = threshold_results.get("opens", [])
    v_open = None
    open_label = None
    open_best_time = None
    
    if opens and opens[0]['threshold_R'] is not None:
        o = opens[0]
        open_label = f"Open {o['location']} (R={o['threshold_R']:.0f}Ω)"
        open_best_time = o['best_time']
        
        # Get component and its nodes
        comp_name = o['location']
        base_name = comp_name.split("_")[0] if "_" in comp_name else comp_name
        comp = circuit.components_dict.get(base_name)
        
        if comp is not None:
            idx_k = getattr(comp, 'idx_1', None) or getattr(comp, 'idx_d', None)
            idx_l = getattr(comp, 'idx_2', None) or getattr(comp, 'idx_s', None)
            
            if idx_k is not None or idx_l is not None:
                xi_kl = build_xi(n, idx_k, idx_l)
                R_branch = o['nominal_R']
                R_eff = o['threshold_R']
                
                # Convert to R_added for the formula
                delta_g = (1.0 / R_eff) - (1.0 / R_branch)
                if abs(delta_g) > 1e-30:
                    R_added = 1.0 / delta_g
                else:
                    R_added = np.inf
                
                v_open = np.zeros(num_steps)
                for t_idx in range(num_steps):
                    lu = result.list_of_lus[t_idx]
                    v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI
                    delta_v = compute_large_change(lu, xi_kl, v, R_added, out_idx)
                    v_open[t_idx] = np.real(v[out_idx] + delta_v)
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Nominal response
    ax.plot(time, v_nominal, 'b-', lw=2, label=f'Nominal V({output_node})')
    
    # Tolerance band
    ax.fill_between(time, v_nominal - delta_F, v_nominal + delta_F,
                     alpha=0.15, color='blue', label=f'±{delta_F}V tolerance')
    
    # Short response
    if v_short is not None:
        ax.plot(time, v_short, 'r--', lw=1.5, label=short_label)
        if short_best_time is not None:
            ax.axvline(x=short_best_time, color='red', ls=':', alpha=0.5)
    
    # Open response
    if v_open is not None:
        ax.plot(time, v_open, 'g-.', lw=1.5, label=open_label)
        if open_best_time is not None:
            ax.axvline(x=open_best_time, color='green', ls=':', alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'Fault Comparison at V({output_node}) | δF = {delta_F}V')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, ls='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)
    print(f"Fault comparison plot saved to {folder}/{name}.png")


def plot_all_faults(circuit, result, output_node, threshold_results, delta_F,
                     folder="./figures/fault", name="all_faults"):
    """
    Plot nominal response with ALL faulted responses (not just top one).
    Each short and open gets its own curve.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    
    tensor = result.sensitivities
    time = tensor.sweep_axis
    num_steps = len(time)
    n = circuit.total_dim
    out_idx = circuit.get_idx(output_node)
    
    # Get nominal V(out)
    v_nominal = np.zeros(num_steps)
    for t_idx in range(num_steps):
        v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI
        v_nominal[t_idx] = np.real(v[out_idx])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Nominal + tolerance band
    ax.plot(time, v_nominal, 'b-', lw=2, label=f'Nominal V({output_node})')
    ax.fill_between(time, v_nominal - delta_F, v_nominal + delta_F,
                     alpha=0.1, color='blue', label=f'±{delta_F}V tolerance')
    
    # Plot each short
    shorts = threshold_results.get("shorts", [])
    for i, s in enumerate(shorts):
        if s['threshold_R'] is None:
            continue
        
        parts = s['location'].split("<->")
        n1_name, n2_name = parts[0], parts[1]
        idx_k = circuit.node_map.get(n1_name)
        idx_l = circuit.node_map.get(n2_name)
        if idx_k is None:
            try: idx_k = circuit.node_map.get(int(n1_name))
            except ValueError: pass
        if idx_l is None:
            try: idx_l = circuit.node_map.get(int(n2_name))
            except ValueError: pass
        
        if idx_k is None and idx_l is None:
            continue
        
        xi_kl = build_xi(n, idx_k, idx_l)
        v_faulted = np.zeros(num_steps)
        for t_idx in range(num_steps):
            lu = result.list_of_lus[t_idx]
            v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI
            delta_v = compute_large_change(lu, xi_kl, v, s['threshold_R'], out_idx)
            v_faulted[t_idx] = np.real(v[out_idx] + delta_v)
        
        ax.plot(time, v_faulted, '--', lw=1.2,
                label=f"Short {s['location']} ({s['threshold_R']:.0f}Ω)")
    
    # Plot each open
    opens = threshold_results.get("opens", [])
    for i, o in enumerate(opens):
        if o['threshold_R'] is None:
            continue
        
        comp_name = o['location']
        base_name = comp_name.split("_")[0] if "_" in comp_name else comp_name
        comp = circuit.components_dict.get(base_name)
        
        if comp is None:
            continue
        
        idx_k = getattr(comp, 'idx_1', None) or getattr(comp, 'idx_d', None)
        idx_l = getattr(comp, 'idx_2', None) or getattr(comp, 'idx_s', None)
        
        if idx_k is None and idx_l is None:
            continue
        
        xi_kl = build_xi(n, idx_k, idx_l)
        R_branch = o['nominal_R']
        R_eff = o['threshold_R']
        
        delta_g = (1.0 / R_eff) - (1.0 / R_branch)
        if abs(delta_g) < 1e-30:
            continue
        R_added = 1.0 / delta_g
        
        v_faulted = np.zeros(num_steps)
        for t_idx in range(num_steps):
            lu = result.list_of_lus[t_idx]
            v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI
            delta_v = compute_large_change(lu, xi_kl, v, R_added, out_idx)
            v_faulted[t_idx] = np.real(v[out_idx] + delta_v)
        
        ax.plot(time, v_faulted, '-.', lw=1.2,
                label=f"Open {o['location']} ({o['threshold_R']:.0f}Ω)")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'All Fault Responses at V({output_node}) | δF = {delta_F}V')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, ls='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)
    print(f"All faults plot saved to {folder}/{name}.png")