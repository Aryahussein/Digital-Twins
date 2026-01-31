from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_sparse
from postprocessing import map_voltages
from assembleYmatrix import generate_stamps
import matplotlib.pyplot as plt
import numpy as np

def dc_nodal_analysis(netlist, w = 0):

    # Convert netlist to components
    components = parse_netlist(netlist)

    # print(components)

    # Set ground as node 0 and re-assign node indices for the rest
    node_index, mna_index, total_dim = build_node_index(components)

    # Build Y matrix and I vector from components
    Y, I = generate_stamps(components, node_index, mna_index, total_dim, w=w)
    # print(Y.todense())
    
    # Show sparsity pattern of Y matrix (optional)
    # plt.spy(Y)
    # plt.show()

    # Use existing sparse matrix solver
    V = solve_sparse(Y, I)

    # 1. Node Voltages
    voltages = map_voltages(V, node_index)
    for node in sorted(voltages):
        val = voltages[node]
        if w == 0:
            print(f"Node {node}: {val.real:.6f} V")
        else:
            mag = np.abs(val)
            phase = np.degrees(np.angle(val))
            print(f"Node {node}: {mag:.6f} V ∠ {phase:.2f}°")

    # 2. Branch Currents (from Voltage Sources & Inductors)
    if mna_index:
        print("\n--- Branch Currents ---")
        sorted_mna = sorted(mna_index.items(), key=lambda item: item[1])
        for name, idx in sorted_mna:
            val = V[idx]
            if w == 0:
                print(f"{name}: {val.real:.6f} A")
            else:
                mag = np.abs(val)
                phase = np.degrees(np.angle(val))
                print(f"{name}: {mag:.6f} A ∠ {phase:.2f}°")
    

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
    node_index, mna_index, total_dim = build_node_index(components)
    
    # 2. Setup Frequency Range (Logarithmic spacing)
    frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)
    magnitudes_db = []
    phases_deg = []
    
    print(f"Running sweep from {start_freq} Hz to {stop_freq} Hz on {netlist_file}...")

    # 3. Frequency Sweep Loop
    for f in frequencies:
        w = 2 * np.pi * f  # Convert Hz to Rad/s
        
        # Build Matrix for this specific frequency w
        Y, I = generate_stamps(components, node_index, mna_index, total_dim, w=w)
        
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
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
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./figures/ac/{name}.png", dpi = 600, bbox_inches = "tight" )

if __name__ == "__main__":
    test_directory = "testfiles/"
    # netlist = "testfiles/test_with_vccs.txt"
    # dc_nodal_analysis(netlist)
    # Example Bode Plot
    run_bode_plot(test_directory + "ac_lowpass.txt", output_node=2, start_freq=10, stop_freq=100000, points=200, name = "lowpass")
    run_bode_plot(test_directory + "ac_resonance.txt", output_node=3, start_freq=10, stop_freq=100000, points=200, name = "resonance")
