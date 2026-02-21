import numpy as np
from sources import evaluate_all_time_sources
from solver import solve_nonlinear_circuit, solve_linear_circuit
from assembleYmatrix import stamp_dynamic_components, stamp_transient_components
from sensitivity import compute_step_sensitivities
from tools import print_solution


def _apply_ac_sources(sources, components, node_map):
    """
    Replace DC source values with AC magnitudes in the source vector.
    For AC analysis, the excitation should come from the 'ac' field, not 'value' (DC).
    """
    ac_sources = sources.copy().astype(complex)
    for name, comp in components.items():
        type_char = name[0].upper()
        ac_mag = comp.get("ac", 0.0)

        if type_char == 'V' and ac_mag != 0.0:
            idx = node_map[name]  # MNA branch variable index
            ac_sources[idx] = ac_mag
        elif type_char == 'I' and ac_mag != 0.0:
            n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
            i = node_map.get(n1)
            j = node_map.get(n2)
            # Undo DC stamp and apply AC magnitude
            dc_val = comp.get("value", 0.0)
            if i is not None:
                ac_sources[i] += dc_val  # undo the -I stamp
                ac_sources[i] -= ac_mag  # apply AC magnitude
            if j is not None:
                ac_sources[j] -= dc_val  # undo the +I stamp
                ac_sources[j] += ac_mag  # apply AC magnitude

    return ac_sources


def run_op(Y, sources, components, node_map, output_nodes=None, sensitivity=False, nonlinear=False, w=0):
    """
    Run DC Operating Point analysis.
    
    Returns:
        VI: solution vector (node voltages and branch currents)
        lu: LU factorization object
        sensitivities: sensitivity dict or None
    """
    stamp_dynamic_components(Y, sources, components, node_map, w=w)

    if nonlinear and w == 0:
        V_guess = np.zeros_like(sources)
        lu, VI = solve_nonlinear_circuit(Y, sources, components, node_map, V_guess,
                                         max_iter=100, tol=1e-9, num_steps=3)
    elif not nonlinear:
        lu, VI = solve_linear_circuit(Y, sources)
    else:
        raise ValueError("Only linear AC analysis is supported")

    print_solution(VI, node_map, w=0.0)

    sensitivities = None
    if sensitivity:
        sensitivities = compute_step_sensitivities(lu, VI, components, node_map, output_nodes, w=0)

    return VI, lu, sensitivities


def run_ac_sweep(Y_base, sources_base, components, node_map,
                 start_freq=10, stop_freq=100000, points=100,
                 output_nodes=None, keep_lus=False, sensitivity=False):
    """
    Run AC small-signal frequency sweep.
    
    Returns:
        frequencies: array of frequency points (Hz)
        VIs: 2D array (num_freq x num_unknowns) of complex phasors
        list_of_lus: list of LU objects (if keep_lus=True)
        sensitivities: list of per-step sensitivity dicts (if sensitivity=True)
    """
    frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)

    list_of_sensitivities_per_freq_step = [] if sensitivity else None
    list_of_lus = [] if keep_lus else None
    VIs = []

    # Apply AC source magnitudes instead of DC values
    ac_sources = _apply_ac_sources(sources_base, components, node_map)

    for f in frequencies:
        w = 2 * np.pi * f

        # Fresh copy of base matrices for this frequency step
        Y_step = Y_base.astype(complex)
        sources_step = ac_sources.copy()

        # Stamp dynamic components (L and C) at this frequency
        Y, sources = stamp_dynamic_components(Y_step, sources_step, components, node_map, w=w)

        # Solve
        lu, VI = solve_linear_circuit(Y, sources)
        VIs.append(VI)

        if keep_lus:
            list_of_lus.append(lu)

        if sensitivity:
            step_sens = compute_step_sensitivities(
                lu, VI, components, node_map,
                output_nodes=output_nodes, w=w
            )
            list_of_sensitivities_per_freq_step.append(step_sens)

    VIs = np.array(VIs)
    return frequencies, VIs, list_of_lus, list_of_sensitivities_per_freq_step


def transient_analysis_loop(Y_base, sources_base, components, node_map,
                            t_stop, dt, output_nodes=None, nonlinear=False,
                            sensitivity=False, keep_lus=False):
    """
    Perform transient analysis using Backward Euler integration.

    Parameters:
        Y_base      : Base admittance matrix (contains only static R, G stamps)
        sources_base: Base source vector (zeros for transient; sources stamped per step)
        components  : Parsed netlist dictionary (with original source definitions)
        node_map    : Variable-to-index mapping
        t_stop      : End time (seconds)
        dt          : Time step (seconds)
        output_nodes: List of nodes for sensitivity computation
        nonlinear   : Whether circuit contains nonlinear elements
        sensitivity : Whether to compute adjoint sensitivity per step
        keep_lus    : Whether to store LU factorizations
        
    Returns:
        time: array of time points
        results: 2D array (num_steps x num_unknowns)
        list_of_lus: list of LU objects or None
        sensitivities: list of per-step sensitivity dicts or None
    """
    num_steps = int(t_stop / dt)
    num_nodes = len(node_map)

    # Time array matches the actual loop computation: t = step * dt
    time = np.array([step * dt for step in range(num_steps)])
    results = np.zeros((num_steps, num_nodes))

    # Initial condition (all zeros unless otherwise specified)
    v_prev = np.zeros(num_nodes)

    list_of_lus = [] if keep_lus else None
    list_of_sensitivities_per_time_step = [] if sensitivity else None

    for step in range(num_steps):
        t = step * dt

        # Evaluate time-dependent sources at current time
        comp_t = evaluate_all_time_sources(components, t)

        # Fresh copy of base matrix (only has R, G; no sources)
        Y_step = Y_base.copy()
        sources_step = sources_base.copy()

        # Stamp sources (with current time values) and BE companion models
        Y_step, sources_step = stamp_transient_components(
            Y_step, sources_step, comp_t, node_map, dt, v_prev
        )

        # Solve system
        if nonlinear:
            lu, VI = solve_nonlinear_circuit(
                Y_step, sources_step, comp_t, node_map, v_prev,
                max_iter=100, tol=1e-9, num_steps=1
            )
        else:
            lu, VI = solve_linear_circuit(Y_step, sources_step)

        if keep_lus:
            list_of_lus.append(lu)

        if sensitivity:
            step_sens = compute_step_sensitivities(lu, VI, components, node_map, output_nodes, dt=dt)
            list_of_sensitivities_per_time_step.append(step_sens)

        # Store results and advance state
        results[step, :] = VI
        v_prev = VI.copy()

    return time, results, list_of_lus, list_of_sensitivities_per_time_step
