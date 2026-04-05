"""
Simulation Plotting and Output Utility Module.

This module provides visualization and console output tools for the EDA simulator. 
It takes `SimulationResult` objects and generates industry-standard Bode plots, 
transient waveforms, DC sweeps, sensitivity curves, and formatted operating 
point summaries.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Must come before importing pyplot for headless environments
import matplotlib.pyplot as plt


def _ensure_dir(folder):
    """Creates the output directory if it doesn't exist."""
    os.makedirs(folder, exist_ok=True)


# ==========================================
# AC PLOTTING (FREQUENCY DOMAIN)
# ==========================================

def make_bode_plot(result, output_nodes, folder="./figures/ac", name="bodeplot"):
    """Plots standard Magnitude (dB) and Phase (degrees) vs Frequency."""
    if not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    
    _ensure_dir(folder)
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
    
    _ensure_dir(folder)
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
    _ensure_dir(folder)
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
    
    _ensure_dir(folder)
    print(f"Plotting transient sensitivity for output node {output_node} w.r.t {target_component}...")
    
    time = result.sweep_axis
    V_out = result.get_voltage(output_node)
    sens_array = result.get_sensitivity(output_node, target_component, output_format="series")

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
    _ensure_dir(folder)
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
    
    _ensure_dir(folder)
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

    V = result.VI

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
