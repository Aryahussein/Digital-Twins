import numpy as np
import matplotlib
matplotlib.use('Agg')  # This must come before importing pyplot
import matplotlib.pyplot as plt

def make_bode_plot(frequencies, VI, node_map, output_node, folder="./figures/ac", name="bodeplot"):
    """Plots standard Magnitude (dB) and Phase (deg) vs Frequency.
    output_node can be a single node or a list of nodes.
    """
    # Normalize to list
    if not isinstance(output_node, (list, tuple)):
        output_node = [output_node]
    
    print(f"Plotting Bode plot for output node(s) {output_node}...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    for node in output_node:
        idx = node_map[node]
        V_out = VI[:, idx] 
        
        mag = np.abs(V_out)
        mag = np.where(mag == 0, 1e-12, mag)
        mag_db = 20 * np.log10(mag)
        phase = np.angle(V_out, deg=True)
        
        ax1.semilogx(frequencies, mag_db, linewidth=2, label=f"Node {node}")
        ax2.semilogx(frequencies, phase, linewidth=2, label=f"Node {node}")
    
    # --- Magnitude Plot ---
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(f"Bode Plot")
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

def plot_ac_sensitivity(frequencies, VI, sensitivities, node_map, output_node, target_component, folder="./figures/ac", name="ac_sensitivity"):
    """Plots Output Magnitude alongside the Sensitivity Magnitude for a specific component.
    output_node can be a single node or a list (uses first element).
    """
    # Use first node if a list is passed
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]
    
    print(f"Plotting AC sensitivity for output node {output_node} w.r.t {target_component}...")
    idx = node_map[output_node]
    V_out = VI[:, idx]
    
    mag = np.abs(V_out)
    mag = np.where(mag == 0, 1e-12, mag)
    mag_db = 20 * np.log10(mag)

    # print(sensitivities)
    
    # Extract sensitivity array from the nested dictionary
    sens_mags = np.abs(sensitivities[output_node][target_component])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    # --- Output Magnitude ---
    ax1.semilogx(frequencies, mag_db, lw=2)
    ax1.set_ylabel("Output Mag (dB)")
    ax1.set_title(f"AC Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # --- Sensitivity Magnitude ---
    ax2.semilogx(frequencies, sens_mags, color='red', lw=2)
    ax2.set_ylabel(f"| dV_{output_node} / d{target_component} |")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_transient(time, VI, node_map, output_node, folder="./figures/tran", name="transient"):
    """Plots standard Voltage vs Time."""
    print(f"Plotting transient response for output node {output_node}...")
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    for node in output_node:
        node_idx = node_map[node]
        V_out = np.real(VI[:, node_idx]) 
        ax.plot(time, V_out, linewidth=2, label=f"Node {node}")

    ax.set_ylabel(f"Voltage (V)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_title(f"Transient Response: Node {output_node}")
    ax.grid(True, ls="--", alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_transient_sensitivity(time, VI, sensitivities, node_map, output_node, target_component, folder="./figures/tran", name="tran_sensitivity"):
    """Plots Transient Voltage alongside the Transient Sensitivity for a specific component."""
    print(f"Plotting transient sensitivity for output node {output_node} w.r.t {target_component}...")
    output_node = output_node[0] 
    idx = node_map[output_node]
    V_out = np.real(VI[:, idx])
    
    # Extract real part of the sensitivity over time
    sens_array = np.real(sensitivities[output_node][target_component])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    # --- Transient Voltage ---
    ax1.plot(time, V_out, 'b-', lw=2)
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Transient Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)

    # --- Transient Sensitivity ---
    ax2.plot(time, sens_array, color='red', lw=2)
    ax2.set_ylabel(f"dV_{output_node} / d{target_component}")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, ls="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def print_solution(V, node_map, w=0.0):
    """
    Prints the solution vector V using the unified node_map.
    Automatically distinguishes between Voltages (node keys) and Currents (string keys).
    """
    print(f"\n--- Simulation Results ({'DC' if w == 0 else f'AC @ {w/(2*np.pi):.2f} Hz'}) ---")
    
    # Separate the map into nodes and MNA components for organized printing
    # (Assuming nodes are stored as integers/strings like 'node_1' 
    # and MNA as component names like 'V1')
    nodes = []
    branches = []
    
    for key, idx in node_map.items():
        if isinstance(key, int): # It's a node number
            nodes.append((key, idx))
        else: # It's an MNA component name (V or L)
            branches.append((key, idx))

    # 1. Print Node Voltages
    print("Node Voltages:")
    for node, idx in sorted(nodes):
        val = V[idx]
        if w == 0:
            print(f"  Node {node}: {val.real:10.6f} V")
        else:
            mag = np.abs(val)
            phase = np.degrees(np.angle(val))
            print(f"  Node {node}: {mag:10.6f} V ∠ {phase:7.2f}°")

    # 2. Print Branch Currents
    if branches:
        print("\nBranch Currents:")
        for name, idx in sorted(branches):
            val = V[idx]
            if w == 0:
                print(f"  {name:7}: {val.real:10.6f} A")
            else:
                mag = np.abs(val)
                phase = np.degrees(np.angle(val))
                print(f"  {name:7}: {mag:10.6f} A ∠ {phase:7.2f}°")

