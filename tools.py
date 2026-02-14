import numpy as np
import matplotlib
matplotlib.use('Agg')  # This must come before importing pyplot
import matplotlib.pyplot as plt
from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_sparse, solve_LU, get_node_and_branch_currents, solve_adjoint
from postprocessing import map_voltages
from assembleYmatrix import generate_stamps


def get_all_sensitivities(components, VI, PsiPhi, node_map, w=0):
    sensitivities = {}

    for name, comp in components.items():
        # 1. Get Indices and branch voltages
        n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
        idx1, idx2 = node_map.get(n1), node_map.get(n2) # get_idx logic

        # Helper to get the voltage difference across a branch
        # (v1 - v2). If a node is ground, its voltage is 0.
        VI_branch = (VI[idx1] if idx1 is not None else 0) - \
                   (VI[idx2] if idx2 is not None else 0)
        
        PsiPhi_branch = (PsiPhi[idx1] if idx1 is not None else 0) - \
                       (PsiPhi[idx2] if idx2 is not None else 0)

        # 2. Apply the sensitivity formula based on component type
        if name.startswith("R"):
            # Sensitivity w.r.t Resistance (R):
            # dY/dR = (1/R^2) * VI_branch * PsiPhi_branch
            R = comp["value"]
            sensitivities[name] = (1.0 / (R**2)) * (VI_branch * PsiPhi_branch)

        elif name.startswith("C"):
            # Sensitivity w.r.t Capacitance (C):
            # dY/dC = j*w. Sensitivity = - (j*w * VI_branch * PsiPhi_branch)
            sensitivities[name] = -1j * w * (VI_branch * PsiPhi_branch)

        elif name.startswith("L"):
            # For Inductors, we use the MNA branch current
            # Row index for the inductor current is its entry in node_map
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            
            # dY/dL for an inductor row is -j*w
            # Sensitivity = - (-j*w * i_L * i_L_hat) = j*w * i_L * i_L_hat
            sensitivities[name] = 1j * w * (i_L * i_L_hat)

        elif name.startswith("G"):
            # VCCS: I = gm * (VI_sense_pos - VI_sense_neg)
            # Sensitive w.r.t transconductance (gm)
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)
            
            v_sense = (VI[idx3] if idx3 is not None else 0) - \
                      (VI[idx4] if idx4 is not None else 0)
            
            # For VCCS, sensitivity = - (PsiPhi_branch_current * VI_sense)
            # In adjoint, this is PsiPhi_branch (voltage at the current source nodes)
            sensitivities[name] = - (PsiPhi_branch * v_sense)

    return sensitivities

def run_bode_plot(netlist_file, output_node, start_freq=10, stop_freq=100000, points=100, name="bodeplot"):
    """
    Runs a frequency sweep and plots Magnitude (dB) and Phase.
    
    netlist_file: Path to the .txt file
    output_node: The integer node ID to plot (e.g., 2)
    start_freq: Start Hz
    stop_freq: Stop Hz
    points: Number of steps
    """
    
    # 1. Parse Netlist Once
    components = parse_netlist(netlist_file)
    node_index, total_dim = build_node_index(components)
    
    # 2. Setup Frequency Range (Logarithmic spacing)
    frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)
    magnitudes_db = []
    phases_deg = []
    
    print(f"Running sweep from {start_freq} Hz to {stop_freq} Hz on {netlist_file}...")

    # 3. Frequency Sweep Loop
    for f in frequencies:
        w = 2 * np.pi * f  # Convert Hz to Rad/s
        
        # Build Matrix for this specific frequency w
        Y, I = generate_stamps(components, node_index, total_dim, w=w)
        
        # Solve
        solution = solve_sparse(Y, I)
        
        # Map Voltages
        voltages = map_voltages(solution, node_index)
        
        # Extract Data for the specific output node
        v_out = voltages.get(output_node, 0.0)
        
        # Convert to dB and Degrees
        mag = np.abs(v_out)
        phase = np.angle(v_out, deg=True)
        
        # Avoid log(0) error
        mag_db = 20 * np.log10(mag) if mag > 1e-12 else -100
        
        magnitudes_db.append(mag_db)
        phases_deg.append(phase)

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    
    # Magnitude Plot
    ax1.semilogx(frequencies, magnitudes_db, 'b-', linewidth=2)
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(f"Bode Plot: Node {output_node}")
    ax1.grid(True, which="both", ls="-", alpha=0.6)
    
    # Mark -3dB point (approx) for convenience
    max_gain = np.max(magnitudes_db)
    ax1.axhline(max_gain - 3, color='r', linestyle='--', alpha=0.5, label="-3dB line")
    ax1.legend()

    # Phase Plot
    ax2.semilogx(frequencies, phases_deg, 'r-', linewidth=2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (Degrees)")
    ax2.set_yticks(np.arange(-180, 181, 45))  # Nice 45-degree steps
    ax2.grid(True, which="both", ls="-", alpha=0.6)
    
    # plt.show()
    fig.savefig(f"./figures/ac/{name}.png", dpi = 600, bbox_inches = "tight" )

def plot_sensitivity_sweep(components, output_node, target_component, start_f=10, end_f=1000, name="sensitiviy"):
    node_map, total_dim = build_node_index(components)
    freqs = np.logspace(np.log10(start_f), np.log10(end_f), 200)
    
    v_out_mags = []
    sens_mags = []

    for f in freqs:
        w = 2 * np.pi * f
        Y, sources = generate_stamps(components, node_map, total_dim, w)
        
        lu = solve_LU(Y)
        VI = get_node_and_branch_currents(lu, sources)
        
        PsiPhi = solve_adjoint(lu, output_node, node_map, total_dim)

        # 3. Calculate Sensitivity for R1
        s = get_all_sensitivities(components, VI, PsiPhi, node_map, w)
        
        v_out_mags.append(np.abs(VI[node_map[output_node]]))
        sens_mags.append(np.abs(s[target_component]))

    v_out_mags = np.array(v_out_mags)
    sens_mags = np.array(sens_mags)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    
    mag_db = 20 * np.log10(v_out_mags)
    ax1.semilogx(freqs, mag_db, lw=2)
    # ax1.set_ylabel("Output Magnitude |Vout| (V)")
    ax1.set_ylabel("Magnitude (dB)")
    # ax1.set_title("Twin-T Notch Filter Response")
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    max_gain = np.max(mag_db)
    print(f"Max Gain: {max_gain:.4f} dB")
    ax1.axhline(max_gain - 3, color='r', linestyle='--', alpha=0.5, label="-3dB line")
    ax1.legend()

    ax2.semilogx(freqs, sens_mags, color='red', lw=2)
    ax2.set_ylabel(f"Sensitivity |dVout / d{target_component}|")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title("Sensitivity vs. Frequency")
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    fig.savefig(f"./figures/ac/{name}.png", dpi = 600, bbox_inches = "tight" )

def print_solution(V, node_map, w=0):
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
            print(f"  Node {node}: {mag:10.6f} V angle {phase:7.2f}deg")

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
                print(f"  {name:7}: {mag:10.6f} A angle {phase:7.2f}deg")

def plot_transient(x_axis, y_axis, name="transient"):
    """
    Plots the transient simulation results.
    
    x_axis: time array
    y_axis: output voltage/current array
   """
    
    x_points = np.array(x_axis)
    y_points = np.array(y_axis)

    plt.scatter(x_points, y_points)

    #plt.show()

    plt.savefig(f"./figures/transient/{name}.png", dpi = 600, bbox_inches = "tight" )