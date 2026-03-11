"""
Simulation Plotting and Output Utility Module.

This module provides visualization and console output tools for the EDA simulator. 
It takes `SimulationResult` objects and generates industry-standard Bode plots, 
transient waveforms, sensitivity curves, and formatted operating point summaries.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come before importing pyplot for headless environments
import matplotlib.pyplot as plt

def make_bode_plot(result, output_nodes, folder="./figures/ac", name="bodeplot"):
    """Plots standard Magnitude (dB) and Phase (degrees) vs Frequency.

    Args:
        result (SimulationResult): The data vault from an AC analysis.
        output_nodes (str or list): A single node name or list of node names to plot.
        folder (str, optional): Output directory. Defaults to "./figures/ac".
        name (str, optional): Output filename. Defaults to "bodeplot".
    """
    if not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    
    print(f"Plotting Bode plot for output node(s) {output_nodes}...")
    frequencies = result.sweep_axis
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    for node in output_nodes:
        # Use our OOP getter! It automatically preserves the complex numbers.
        V_out = result.get_voltage(node) 
        
        mag = np.abs(V_out)
        mag = np.where(mag == 0, 1e-12, mag) # Prevent log(0) warnings
        mag_db = 20 * np.log10(mag)
        phase = np.angle(V_out, deg=True)
        
        ax1.semilogx(frequencies, mag_db, linewidth=2, label=f"Node {node}")
        ax2.semilogx(frequencies, phase, linewidth=2, label=f"Node {node}")
    
    # --- Magnitude Plot ---
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Bode Plot")
    ax1.grid(True, which="both", ls="-", alpha=0.6)
    ax1.legend()

    # --- Phase Plot ---
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (Degrees)")
    ax2.set_yticks(np.arange(-180, 181, 45))
    ax2.grid(True, which="both", ls="-", alpha=0.6)
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_ac_sensitivity(result, output_node, target_component, folder="./figures/ac", name="ac_sensitivity"):
    """Plots Output Magnitude alongside the Adjoint Sensitivity Magnitude.

    Args:
        result (SimulationResult): The data vault from an AC analysis.
        output_node (str or list): The objective node (uses first element if list).
        target_component (str): The parameter to plot sensitivities for (e.g., 'R1').
        folder (str, optional): Output directory. Defaults to "./figures/ac".
        name (str, optional): Output filename. Defaults to "ac_sensitivity".
    """
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]
    
    print(f"Plotting AC sensitivity for output node {output_node} w.r.t {target_component}...")
    
    frequencies = result.sweep_axis
    V_out = result.get_voltage(output_node)
    
    mag = np.abs(V_out)
    mag = np.where(mag == 0, 1e-12, mag)
    mag_db = 20 * np.log10(mag)

    # Use our OOP getter to fetch the complex sensitivity array safely!
    raw_sens = result.get_sensitivity(output_node, target_component)
    sens_mags = np.abs(raw_sens)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    ax1.semilogx(frequencies, mag_db, lw=2)
    ax1.set_ylabel("Output Mag (dB)")
    ax1.set_title(f"AC Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    ax2.semilogx(frequencies, sens_mags, color='red', lw=2)
    ax2.set_ylabel(f"| dV_{output_node} / d{target_component} |")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_transient(result, output_nodes=None, folder="./figures/tran", name="transient"):
    """Plots standard Voltage vs Time waveforms.

    

    Args:
        result (SimulationResult): The data vault from a Transient analysis.
        output_nodes (str, list, or None): Node(s) to plot. If None, plots all nodes.
        folder (str, optional): Output directory. Defaults to "./figures/tran".
        name (str, optional): Output filename. Defaults to "transient".
    """
    print(f"Plotting transient response...")
    
    time = result.sweep_axis
    fig, ax = plt.subplots(figsize=(6, 3))

    if output_nodes is None:
        # Filter out MNA branch currents for standard voltage plots
        output_nodes = [k for k in result.node_map.keys() if not str(k).upper().startswith(('V', 'L', 'E', 'H', 'F'))]
    elif not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    
    for node in output_nodes:
        # OOP getter automatically applies np.real() for transient!
        V_out = result.get_voltage(node) 
        ax.plot(time, V_out, linewidth=2, label=f"Node {node}")

    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8, bbox_to_anchor=(1, 1))
    ax.set_title(f"Transient Response")
    ax.grid(True, ls="--", alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_transient_sensitivity(result, output_node, target_component, folder="./figures/tran", name="tran_sensitivity"):
    """Plots Transient Voltage alongside the Transient Sensitivity series over time.

    Args:
        result (SimulationResult): The data vault from a Transient analysis.
        output_node (str or list): The objective node.
        target_component (str): The parameter to plot sensitivities for.
        folder (str, optional): Output directory. Defaults to "./figures/tran".
        name (str, optional): Output filename. Defaults to "tran_sensitivity".
    """
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0] 
        
    print(f"Plotting transient sensitivity for output node {output_node} w.r.t {target_component}...")
    
    time = result.sweep_axis
    V_out = result.get_voltage(output_node)

    # Use OOP getter to extract the raw backward-traveling time series
    sens_array = result.get_sensitivity(output_node, target_component, output_format="series")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    ax1.plot(time, V_out, 'b-', lw=2)
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Transient Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)

    ax2.plot(time, sens_array, color='red', lw=2)
    ax2.set_ylabel(f"dV_{output_node} / d{target_component}")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, ls="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def print_solution(result):
    """Prints the operating point solution vector nicely to the console.
    
    Automatically distinguishes between Voltages (nodes) and Currents (MNA branches).

    Args:
        result (SimulationResult): A single-point (.OP) data vault.
    """
    is_ac = result.type == ".AC"
    header = f"AC @ {result.sweep_axis[0]:.2e} Hz" if is_ac else "DC / OP"
    print(f"\n--- Simulation Results ({header}) ---")
    
    nodes = []
    branches = []
    
    # Bug Fix: Instead of checking type (int vs str), check standard SPICE naming prefixes
    mna_prefixes = ('V', 'L', 'E', 'H', 'F')
    
    for key, idx in result.node_map.items():
        if str(key).upper().startswith(mna_prefixes):
            branches.append((key, idx))
        else:
            nodes.append((key, idx))

    # We use result.VI directly since this is an OP format
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
