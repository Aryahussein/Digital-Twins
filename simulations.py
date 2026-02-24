from sources import evaluate_all_time_sources
from solver import solve_nonlinear_circuit, solve_linear_circuit
from assembleYmatrix import stamp_dynamic_components, stamp_transient_components
from sensitivity import compute_step_sensitivities
import numpy as np
from tools import print_solution

def run_op(Y, sources, components, node_map, output_nodes=None, sensitivity=False, nonlinear=False, w=0):
    print("freq", w)
    stamp_dynamic_components(Y, sources, components, node_map, w=w)

    if nonlinear and w == 0:
        V_guess = np.zeros_like(sources)
        max_iter = 100
        tol = 1e-9
        num_ramp_steps = 3
        lu, VI = solve_nonlinear_circuit(Y, sources, components, node_map, V_guess, max_iter=max_iter, tol=tol, num_steps=num_ramp_steps)
    elif not nonlinear:
        lu, VI = solve_linear_circuit(Y, sources)
    else:
        raise ValueError("Only linear AC analysis is supported")

    print_solution(VI, node_map, w=0.0)

    sensitivities = None
    if sensitivity:
        sensitivities = compute_step_sensitivities(lu, VI, components, node_map, output_nodes, w=0)

    return VI, lu, sensitivities

import numpy as np

def run_ac_sweep(Y_base, sources_base, components, node_map, start_freq=10, stop_freq=100000, points=100, output_nodes=None, keep_lus=False, sensitivity=False):

    frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)
    total_dim = len(node_map) # Needed for the sensitivity solver
    
    list_of_senstivities_per_freq_step = [] if sensitivity else None
    list_of_lus = [] if keep_lus else None
    VIs = []

    # Frequency Sweep Loop
    for f in frequencies:
        w = 2 * np.pi * f  # Convert Hz to Rad/s

        # 1. ALWAYS create a fresh copy of the base matrices for this frequency step!
        Y_step = Y_base.astype(complex)
        sources_step = sources_base.astype(complex)

        # 2. Stamp dynamic components (L and C) using the specific w
        Y, sources = stamp_dynamic_components(Y_step, sources_step, components, node_map, w=w)

        # 3. Solve the linear circuit (Note: Y and sources are complex matrices now)
        lu, VI = solve_linear_circuit(Y, sources)
        VIs.append(VI)

        # 4. Store LU factorization if requested
        if keep_lus:
            list_of_lus.append(lu)

        # 5. Calculate Sensitivities (passing missing 'components' and 'total_dim')
        if sensitivity:
            step_sens = compute_step_sensitivities(
                lu, VI, components, node_map, 
                output_nodes=output_nodes, w=w
            )
            list_of_senstivities_per_freq_step.append(step_sens)
        
    VIs = np.array(VIs)

    return frequencies, VIs, list_of_lus, list_of_senstivities_per_freq_step

def transient_analysis_loop(Y_base, sources_base, components, node_map, t_stop, dt, output_nodes = None, nonlinear=False, sensitivity=False, keep_lus=False):
    """
    Perform transient analysis using Backward Euler.

    Parameters:
        components  : parsed netlist dictionary
        t_stop      : end time
        dt          : timestep
        num_nodes   : number of non-ground nodes
        stamping functions must be provided
    """
    num_steps = int(t_stop / dt)
    num_nodes = len(node_map)

    # Storage for results
    time = np.linspace(0, t_stop, num_steps)
    results = np.zeros((num_steps, num_nodes))

    # initial condition (all zeros unless otherwise specified)
    v_prev = np.zeros(num_nodes)
    sources_prev = np.zeros(num_nodes)

    list_of_lus = None
    if keep_lus:
        list_of_lus = []

    list_of_senstivities_per_time_step = None
    if sensitivity:
        list_of_senstivities_per_time_step = []

    for step in range(num_steps):

        t = step * dt
        # print(f"dt = {dt}, t = {t}")

        # replace time-dependent sources with dc equivalents
        comp_t = evaluate_all_time_sources(components, t)
        # print("comp_t", comp_t)
        Y_step = Y_base.copy()
        sources_step = sources_base.copy()

        Y_step, sources_step = stamp_transient_components(Y_step, sources_step, comp_t, node_map, dt, v_prev)
        # print(Y_step.todense())

        # Solve system
        if nonlinear:
            max_iter = 100
            tol = 1e-9
            num_ramp_steps = 1
            lu, VI = solve_nonlinear_circuit(Y_step, sources_step, comp_t, node_map, v_prev, max_iter=max_iter, tol=tol, num_steps=num_ramp_steps)
        else:
            lu, VI = solve_linear_circuit(Y_step, sources_step)

        if keep_lus:
            list_of_lus.append(lu)

        if sensitivity:
            list_of_senstivities = compute_step_sensitivities(lu, VI, components, node_map, output_nodes)
            list_of_senstivities_per_time_step.append(list_of_senstivities)

        # Store results
        results[step, :] = VI

        # Update state for next timestep
        v_prev = VI.copy()

    return time, results, list_of_lus, list_of_senstivities_per_time_step
