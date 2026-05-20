"""
Simulation Plotting and Output Utility Module.

This module provides visualization and console output tools for the EDA simulator.
It takes `SimulationResult` objects and generates industry-standard Bode plots,
transient waveforms, DC sweeps, sensitivity curves, and formatted operating
point summaries.
"""

import numpy as np
import matplotlib
import os

matplotlib.use("Agg")  # Must come before importing pyplot for headless environments
import matplotlib.pyplot as plt
from applications.large_change_sensitivity import build_xi, compute_large_change
from scipy.stats import norm

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


def plot_ac_sensitivity(
    result, output_node, target_component, folder="./figures/ac", name="ac_sensitivity"
):
    """Plots Output Magnitude alongside the Adjoint Sensitivity Magnitude."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]

    print(
        f"Plotting AC sensitivity for output node {output_node} w.r.t {target_component}..."
    )

    frequencies = result.sweep_axis
    V_out = result.get_voltage(output_node)

    mag = np.abs(V_out)
    mag = np.where(mag == 0, 1e-12, mag)
    mag_db = 20 * np.log10(mag)

    raw_sens = result.get_sensitivity(output_node, target_component)
    sens_mags = np.abs(raw_sens)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.semilogx(frequencies, mag_db, lw=2, color="blue")
    ax1.set_ylabel("Output Mag (dB)")
    ax1.set_title(f"AC Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    ax2.semilogx(frequencies, sens_mags, lw=2, color="red")
    ax2.set_ylabel(f"| dV_{output_node} / d{target_component} |")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)


# ==========================================
# TRANSIENT PLOTTING (TIME DOMAIN)
# ==========================================

def plot_transient(
    result, output_nodes=None, folder="./figures/tran", name="transient", mark_step=None
):
    """Plots standard Voltage vs Time waveforms, with an optional evaluation marker."""
    print(f"Plotting transient response...")

    time = result.sweep_axis
    fig, ax = plt.subplots(figsize=(6, 4))

    if output_nodes is None:
        output_nodes = [
            k
            for k in result.node_map.keys()
            if not str(k).upper().startswith(("V", "L", "E", "H", "F"))
        ]
    elif not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]

    for node in output_nodes:
        V_out = result.get_voltage(node)
        ax.plot(time, V_out, lw=2, label=f"Node {node}")

    # --- NEW: Add the vertical marker for Yield Analysis Evaluation ---
    if mark_step is not None:
        eval_time = time[mark_step]
        ax.axvline(eval_time, color='black', linestyle=':', lw=2, 
                   label=f"Yield Eval ({eval_time:.2e} s)")

    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Transient Response")
    ax.grid(True, ls="--", alpha=0.6)
    
    # Put legend outside if it gets crowded, or keep best
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_transient_sensitivity_corrected_units(
    circuit,           # <-- NEW: Required to extract nominal physical values
    result,
    output_node,
    target_component,
    folder="./figures/tran",
    name="tran_sensitivity",
    mark_step=None,
    delta_pct=0.05     # <-- NEW: Default 5% perturbation
):
    """Plots Transient Voltage alongside Normalized Transient Sensitivities (Voltage Shift)."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]
        
    if isinstance(target_component, str):
        target_components = [target_component]
    else:
        target_components = target_component

    print(f"Plotting normalized transient sensitivity for V({output_node}) w.r.t {target_components}...")

    time = result.sweep_axis
    V_out = result.get_voltage(output_node)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Top Plot: Primal Voltage
    ax1.plot(time, V_out, lw=2, color="blue", label=f"Nominal V({output_node})")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Transient Sensitivities: Node {output_node}")
    ax1.grid(True, ls="--", alpha=0.6)

    # Bottom Plot: NORMALIZED Sensitivities
    for param in target_components:
        raw_sens = result.get_sensitivity(output_node, param)
        
        # 1. Extract the base component name (e.g., 'M1' from 'M1_W')
        comp_name = param.split("_")[0]
        comp = next((c for c in circuit.components if c.name == comp_name), None)
        
        if comp is not None:
            # 2. Extract nominal value and calculate physical delta
            p_nom = comp.get_nominal_value(param)
            delta_p = p_nom * delta_pct
            
            # 3. Normalize: (dV / dp) * delta_p = Voltage Shift (V)
            norm_sens = raw_sens * delta_p
            label = f"ΔV for {delta_pct*100}% Δ{param}"
        else:
            # Fallback if component lookup fails
            norm_sens = raw_sens
            label = f"d(V)/d({param}) [Raw]"

        ax2.plot(time, norm_sens, lw=2, label=label)

    # Now the Y-axis is pure Volts!
    ax2.set_ylabel(f"Voltage Shift (V)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, ls="--", alpha=0.6)
    
    if mark_step is not None:
        eval_time = time[mark_step]
        ax1.axvline(eval_time, color='black', linestyle=':', lw=2, label=f"Max Sens ({eval_time:.2e} s)")
        ax2.axvline(eval_time, color='black', linestyle=':', lw=2)

    ax1.legend(loc="best", fontsize=9)
    ax2.legend(loc="best", fontsize=8, ncol=min(3, len(target_components)))

    fig.tight_layout()
    os.makedirs(folder, exist_ok=True)
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)


def plot_transient_sensitivity(
    result,
    output_node,
    target_component,
    folder="./figures/tran",
    name="tran_sensitivity",
):
    """Plots Transient Voltage alongside the Transient Sensitivity series over time."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]

    print(
        f"Plotting transient sensitivity for output node {output_node} w.r.t {target_component}..."
    )

    time = result.sweep_axis
    V_out = result.get_voltage(output_node)
    sens_array = result.get_sensitivity(output_node, target_component)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(time, V_out, lw=2, color="blue")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Transient Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)

    ax2.plot(time, sens_array, lw=2, color="red")
    ax2.set_ylabel(f"dV_{output_node} / d{target_component}")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, ls="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)

def plot_transient_sensitivity_to_radiation(
    result,
    output_node,
    target_component,
    radiation,
    folder="./figures/tran",
    name="tran_sensitivity",
):
    """Plots Transient Voltage alongside the Transient Sensitivity series over time."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]
    print(
        f"Plotting transient sensitivity for output node {output_node} w.r.t {target_component}..."
    )

    time = result.sweep_axis
    V_out = result.get_voltage(output_node)
    sens_array = result.get_sensitivity(output_node, target_component)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.plot(time, V_out, lw=2, color="blue")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Transient Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)
    for qrad in radiation:
        ax2.semilogy(
            time,
            abs(result.get_vhats(output_node) * qrad / (time[1] - time[0])),
            lw=2,
            label=f"qrad = {qrad:.2e} C",
        )
    ax2.set_ylim(bottom=1e-20, top=17)
    ax2.set_ylabel(f"dV_{output_node}")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)


def plot_integrated_time_series(
    result,
    output_node,
    target_component,
    folder="./figures/tran",
    name="tran_sensitivity",
):
    """Plots Transient Voltage alongside the Transient Sensitivity series over time."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]

    print(
        f"Plotting transient sensitivity for output node {output_node} w.r.t {target_component}..."
    )

    time = result.sweep_axis
    V_out = result.get_voltage(output_node)
    sens_array = result.get_sensitivity(output_node, target_component)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.plot(time, V_out, lw=2, color="blue")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Transient Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)

    ax2.plot(time, sens_array, lw=2, color="red")
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
        output_nodes = [
            k
            for k in result.node_map.keys()
            if not str(k).upper().startswith(("V", "L", "E", "H", "F"))
        ]
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


def plot_dc_sensitivity(
    result, output_node, target_component, folder="./figures/dc", name="dc_sensitivity"
):
    """Plots DC Sweep Voltage alongside the DC Sensitivity series."""
    if isinstance(output_node, (list, tuple)):
        output_node = output_node[0]

    print(
        f"Plotting DC sensitivity for output node {output_node} w.r.t {target_component}..."
    )

    sweep_v = result.sweep_axis
    V_out = result.get_voltage(output_node)
    sens_array = result.get_sensitivity(output_node, target_component)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.plot(sweep_v, V_out, lw=2, color="blue")
    ax1.set_ylabel("Output (V/A)")
    ax1.set_title(f"DC Sensitivity: Node {output_node} w.r.t {target_component}")
    ax1.grid(True, ls="--", alpha=0.6)

    ax2.plot(sweep_v, sens_array, lw=2, color="red")
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

    mna_prefixes = ("V", "L", "E", "H", "F")

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


def plot_fault_comparison(
    circuit,
    result,
    output_node,
    threshold_results,
    delta_F,
    folder="./figures/fault",
    name="fault_comparison",
):
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

    if shorts and shorts[0]["threshold_R"] is not None:
        s = shorts[0]
        short_label = f"Short {s['location']} (R={s['threshold_R']:.0f}Ω)"
        short_best_time = s["best_time"]

        # Parse node indices
        parts = s["location"].split("<->")
        n1_name, n2_name = parts[0], parts[1]
        idx_k = circuit.node_map.get(n1_name)
        idx_l = circuit.node_map.get(n2_name)
        if idx_k is None:
            try:
                idx_k = circuit.node_map.get(int(n1_name))
            except ValueError:
                pass
        if idx_l is None:
            try:
                idx_l = circuit.node_map.get(int(n2_name))
            except ValueError:
                pass

        if idx_k is not None or idx_l is not None:
            xi_kl = build_xi(n, idx_k, idx_l)
            R_short = s["threshold_R"]

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

    if opens and opens[0]["threshold_R"] is not None:
        o = opens[0]
        open_label = f"Open {o['location']} (R={o['threshold_R']:.0f}Ω)"
        open_best_time = o["best_time"]

        # Get component and its nodes
        comp_name = o["location"]
        base_name = comp_name.split("_")[0] if "_" in comp_name else comp_name
        comp = circuit.components_dict.get(base_name)

        if comp is not None:
            idx_k = getattr(comp, "idx_1", None) or getattr(comp, "idx_d", None)
            idx_l = getattr(comp, "idx_2", None) or getattr(comp, "idx_s", None)

            if idx_k is not None or idx_l is not None:
                xi_kl = build_xi(n, idx_k, idx_l)
                R_branch = o["nominal_R"]
                R_eff = o["threshold_R"]

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
    ax.plot(time, v_nominal, "b-", lw=2, label=f"Nominal V({output_node})")

    # Tolerance band
    ax.fill_between(
        time,
        v_nominal - delta_F,
        v_nominal + delta_F,
        alpha=0.15,
        color="blue",
        label=f"±{delta_F}V tolerance",
    )

    # Short response
    if v_short is not None:
        ax.plot(time, v_short, "r--", lw=1.5, label=short_label)
        if short_best_time is not None:
            ax.axvline(x=short_best_time, color="red", ls=":", alpha=0.5)

    # Open response
    if v_open is not None:
        ax.plot(time, v_open, "g-.", lw=1.5, label=open_label)
        if open_best_time is not None:
            ax.axvline(x=open_best_time, color="green", ls=":", alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Fault Comparison at V({output_node}) | δF = {delta_F}V")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)
    print(f"Fault comparison plot saved to {folder}/{name}.png")


def plot_all_faults(
    circuit,
    result,
    output_node,
    threshold_results,
    delta_F,
    folder="./figures/fault",
    name="all_faults",
):
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
    ax.plot(time, v_nominal, "b-", lw=2, label=f"Nominal V({output_node})")
    ax.fill_between(
        time,
        v_nominal - delta_F,
        v_nominal + delta_F,
        alpha=0.1,
        color="blue",
        label=f"±{delta_F}V tolerance",
    )

    # Plot each short
    shorts = threshold_results.get("shorts", [])
    for i, s in enumerate(shorts):
        if s["threshold_R"] is None:
            continue

        parts = s["location"].split("<->")
        n1_name, n2_name = parts[0], parts[1]
        idx_k = circuit.node_map.get(n1_name)
        idx_l = circuit.node_map.get(n2_name)
        if idx_k is None:
            try:
                idx_k = circuit.node_map.get(int(n1_name))
            except ValueError:
                pass
        if idx_l is None:
            try:
                idx_l = circuit.node_map.get(int(n2_name))
            except ValueError:
                pass

        if idx_k is None and idx_l is None:
            continue

        xi_kl = build_xi(n, idx_k, idx_l)
        v_faulted = np.zeros(num_steps)
        for t_idx in range(num_steps):
            lu = result.list_of_lus[t_idx]
            v = result.VI[t_idx] if result.VI.ndim == 2 else result.VI
            delta_v = compute_large_change(lu, xi_kl, v, s["threshold_R"], out_idx)
            v_faulted[t_idx] = np.real(v[out_idx] + delta_v)

        ax.plot(
            time,
            v_faulted,
            "--",
            lw=1.2,
            label=f"Short {s['location']} ({s['threshold_R']:.0f}Ω)",
        )

    # Plot each open
    opens = threshold_results.get("opens", [])
    for i, o in enumerate(opens):
        if o["threshold_R"] is None:
            continue

        comp_name = o["location"]
        base_name = comp_name.split("_")[0] if "_" in comp_name else comp_name
        comp = circuit.components_dict.get(base_name)

        if comp is None:
            continue

        idx_k = getattr(comp, "idx_1", None) or getattr(comp, "idx_d", None)
        idx_l = getattr(comp, "idx_2", None) or getattr(comp, "idx_s", None)

        if idx_k is None and idx_l is None:
            continue

        xi_kl = build_xi(n, idx_k, idx_l)
        R_branch = o["nominal_R"]
        R_eff = o["threshold_R"]

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

        ax.plot(
            time,
            v_faulted,
            "-.",
            lw=1.2,
            label=f"Open {o['location']} ({o['threshold_R']:.0f}Ω)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"All Fault Responses at V({output_node}) | δF = {delta_F}V")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{folder}/{name}.png", dpi=600)
    plt.close(fig)
    print(f"All faults plot saved to {folder}/{name}.png")

def plot_worst_case_corners(lc_data, target_node, spec_min, spec_max, tolerance_pct, folder, name, step_idx=-1):
    os.makedirs(folder, exist_ok=True)
    
    alpha_sweep = lc_data.alpha_axis
    v_out = lc_data(node=target_node, step_idx=step_idx)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.fill_between(alpha_sweep * 100, spec_min, spec_max, color='#d4edda', alpha=0.5, label="Passing Spec Region")
    ax.plot(alpha_sweep * 100, v_out, linewidth=3, color='#1f77b4', label="Worst-Case Output Voltage")

    ax.axhline(spec_max, color='#d62728', linestyle='--', linewidth=2, label=f"Spec Max ({spec_max}V)")
    ax.axhline(spec_min, color='#d62728', linestyle='--', linewidth=2, label=f"Spec Min ({spec_min}V)")

    nominal_idx = np.argmin(np.abs(alpha_sweep))
    ax.plot(0, v_out[nominal_idx], 'ko', markersize=8, zorder=5, label=f"Nominal Design ({v_out[nominal_idx]:.2f}V)")

    tol_pos_idx = np.argmin(np.abs(alpha_sweep - tolerance_pct))
    tol_neg_idx = np.argmin(np.abs(alpha_sweep - (-tolerance_pct)))

    ax.plot(tolerance_pct * 100, v_out[tol_pos_idx], 'mo', markersize=9, zorder=5, label=f"Worst-Case Corner (+{tolerance_pct*100:.0f}%)")
    ax.plot(-tolerance_pct * 100, v_out[tol_neg_idx], 'ro', markersize=9, zorder=5, label=f"Worst-Case Corner (-{tolerance_pct*100:.0f}%)")

    ax.axvline(tolerance_pct * 100, color='gray', linestyle=':', linewidth=1.5)
    ax.axvline(-tolerance_pct * 100, color='gray', linestyle=':', linewidth=1.5)

    ax.set_title("Sensitivity-Directed Worst-Case (SDWC) Analysis", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Global Parameter Variation (α %)", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"V({target_node}) [V]", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, framealpha=0.9)

    if v_out[tol_neg_idx] >= spec_min and v_out[tol_pos_idx] <= spec_max:
        status_text = "STATUS: PASS (100% Guaranteed Yield)"
        text_color = 'green'
    else:
        status_text = "STATUS: FAIL (Boundary exceeds spec)"
        text_color = 'red'

    ax.text(0.02, 0.04, status_text, transform=ax.transAxes, fontsize=12, 
            fontweight='bold', color=text_color,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=text_color, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    fig.savefig(os.path.join(folder, f"{name}.png"), dpi=600)
    plt.close()

def plot_combined_yield_pdf(mean_out, sigma_out, spec_min, spec_max, target_node, step_idx, folder, name):
    os.makedirs(folder, exist_ok=True)
    x_axis = np.linspace(mean_out - 6*sigma_out, mean_out + 6*sigma_out, 1000)
    pdf_values = norm.pdf(x_axis, mean_out, sigma_out)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, pdf_values, color='black', linewidth=2, label=f'Woodbury PDF (σ = {sigma_out*1000:.2f}mV)')
    
    pass_region = (x_axis >= spec_min) & (x_axis <= spec_max)
    plt.fill_between(x_axis, pdf_values, where=pass_region, color='green', alpha=0.2, label='Passing Yield')

    fail_low, fail_high = (x_axis < spec_min), (x_axis > spec_max)
    plt.fill_between(x_axis, pdf_values, where=fail_low, color='red', alpha=0.4, label='Failure (Out of Spec)')
    plt.fill_between(x_axis, pdf_values, where=fail_high, color='red', alpha=0.4)

    plt.axvline(spec_min, color='red', linestyle='--', linewidth=2)
    plt.axvline(spec_max, color='red', linestyle='--', linewidth=2)
    plt.axvline(mean_out, color='black', linestyle=':', linewidth=1.5)

    prob_passing = norm.cdf(spec_max, mean_out, sigma_out) - norm.cdf(spec_min, mean_out, sigma_out)
    
    plt.title(f'Rigorous Yield Distribution at Node {target_node}\nYield: {prob_passing*100:.6f}% | DPMO: {(1-prob_passing)*1e6:.1f}', fontweight='bold')
    plt.xlabel('Voltage (V)', fontweight='bold')
    plt.ylabel('Probability Density', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{folder}/{name}.png", dpi=300)
    plt.close()

def plot_unified_pareto(step_specific_ranking, woodbury_deltas, target_node, step_idx, folder, name, combined_adjoint_v=None, combined_woodbury_v=None):
    os.makedirs(folder, exist_ok=True)

    clean_data = [d for d in step_specific_ranking if d['dv_expected'] > 1e-9]
    clean_data.reverse()

    params = [d['param'] for d in clean_data]
    
    adj_vals_mv = [d['dv_expected'] * 1000 for d in clean_data] 
    
    wb_vals_mv = []
    for p in params:
        if p in woodbury_deltas:
            wb_vals_mv.append(np.abs(woodbury_deltas[p]) * 1000)
        else:
            wb_vals_mv.append(0.0)

    if combined_woodbury_v is not None and combined_woodbury_v > 0:
        params.append("COMBINED CORNERS (1σ)")
        adj_vals_mv.append(combined_adjoint_v * 1000)
        wb_vals_mv.append(combined_woodbury_v * 1000)

    dynamic_height = max(6, len(params) * 0.5)
    fig, ax = plt.subplots(figsize=(11, dynamic_height))
    
    y = np.arange(len(params))
    height = 0.4 

    rects1 = ax.barh(y + height/2, adj_vals_mv, height, label='Adjoint Prediction (Linear)', color='lightblue', edgecolor='black')
    rects2 = ax.barh(y - height/2, wb_vals_mv, height, label='Woodbury Truth (Non-Linear)', color='coral', edgecolor='black')

    ax.set_yticks(y)
    ax.set_yticklabels(params, fontweight='bold')
    ax.set_xlabel('Absolute Output Voltage Shift (mV) per 1σ Variation', fontsize=11, fontweight='bold')
    ax.set_title(f'Unified Sensitivity Ranking: V({target_node}) at Step {step_idx}', fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    ax.legend(loc='lower right', framealpha=0.9)

    max_val = max(max(adj_vals_mv), max(wb_vals_mv))
    
    for rect in rects1:
        width = rect.get_width()
        ax.text(width + (max_val * 0.01), rect.get_y() + rect.get_height()/2, 
                f'{width:.2f} mV', va='center', fontweight='bold', color='black', fontsize=9)

    for rect in rects2:
        width = rect.get_width()
        if width > 0.0:
            ax.text(width + (max_val * 0.01), rect.get_y() + rect.get_height()/2, 
                    f'{width:.2f} mV', va='center', fontweight='bold', color='darkred', fontsize=9)

    ax.set_xlim(0, max_val * 1.15)

    plt.tight_layout()
    filepath = os.path.join(folder, f"{name}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_individual_sensitivities(result, all_dv_waveforms, folder, name):
    """Plots absolute 1-sigma voltage shifts for every parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = result.sweep_axis * 1e9 if result.analysis_type == ".TRAN" else result.sweep_axis
    
    for param, wave in all_dv_waveforms.items():
        ax.plot(x_axis, wave * 1000, label=param, alpha=0.7, linewidth=1.5)

    ax.set_ylabel('Absolute Output Shift $\Delta V_{out}$ (mV)', fontweight='bold')
    ax.set_xlabel('Time (ns)' if result.analysis_type == ".TRAN" else 'Sweep', fontweight='bold')
    ax.set_title('Absolute 1$\sigma$ Adjoint Impact ($S_{raw} \cdot \sigma_p$)', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{name}.png"), dpi=300); plt.close()

def plot_sigma_metrics(result, m1_wave, m2_wave, m3_wave, m4_wave, step_m1, step_m2, step_m3, step_m4, folder, name):
    """Compares metrics and marks exactly where evaluation steps occur."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = result.sweep_axis * 1e9 if result.analysis_type == ".TRAN" else result.sweep_axis

    # Unit conversion to mV
    data = [m1_wave*1000, np.sqrt(m2_wave)*1000, m3_wave*1000, m4_wave*1000]
    steps = [step_m1, step_m2, step_m3, step_m4]
    colors = ['royalblue', 'forestgreen', 'grey', 'darkorange']
    labels = ['M1: Max Indiv (1$\sigma$)', 'M2: Total RSS (1$\sigma$)', 'M3: Phys Bounds', 'M4: Woodbury (1$\sigma$)']

    for wave, step, color, label in zip(data, steps, colors, labels):
        ax.plot(x_axis, wave, color=color, linewidth=2, label=label, alpha=0.3 if color=='grey' else 0.8)
        # Mark the actual evaluation point on the curve
        ax.plot(x_axis[step], wave[step], 'o', color=color, markersize=8, markeredgecolor='black', zorder=5)

    ax.set_ylabel('Voltage Deviation (mV)', fontweight='bold')
    ax.set_title('Direct Comparison of Yield Metrics (Markers at Eval Steps)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{name}.png"), dpi=300); plt.close()

def plot_transient_envelope(base_result, lc_1sigma_results, target_node, evaluation_list, folder, name):
    """Master Envelope: Strictly 1-sigma with separate lines for every metric."""
    t_axis = base_result.sweep_axis * 1e9  
    n_idx = base_result.node_map[target_node]
    v_nom = base_result.VI[:, n_idx]
    
    # Use M4 (1-sigma) for the shading
    all_waves = lc_1sigma_results.data[:, :, n_idx]
    v_min, v_max = np.min(all_waves, axis=0), np.max(all_waves, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(t_axis, v_min, v_max, color='red', alpha=0.1, label='Woodbury 1$\sigma$ Envelope')
    ax.plot(t_axis, v_nom, color='black', linewidth=1.5, label='Nominal')

    metric_colors = {'M1': 'blue', 'M2': 'green', 'M3': 'red', 'M4': 'darkorange'}
    for (step, label) in evaluation_list:
        color = metric_colors.get(label[:2], 'purple')
        ax.axvline(t_axis[step], color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.plot(t_axis[step], v_max[step], 'X', color=color, markersize=7, label=f'Eval {label}')
        ax.plot(t_axis[step], v_min[step], 'X', color=color, markersize=7)

    ax.set_title(f"Master 1$\sigma$ Yield Envelope: {target_node}", fontweight='bold')
    ax.set_ylabel("Voltage (V)"); ax.set_xlabel("Time (ns)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{name}.png"), dpi=300); plt.close()

