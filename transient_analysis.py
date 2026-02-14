from assembleYmatrix import generate_stamps_transient
import numpy as np
from sources import evaluate_all_time_sources
from solver import solve_sparse
from node_index import build_node_index
from txt2dictionary import parse_netlist
import matplotlib.pyplot as plt

def transient_analysis_loop(components, t_stop, dt, method):
    """
    Perform transient analysis using Backward Euler.

    Parameters:
        components  : parsed netlist dictionary
        t_stop      : end time
        dt          : timestep
        num_nodes   : number of non-ground nodes
        stamping functions must be provided
    """
    node_map, num_nodes = build_node_index(components)
    num_steps = int(t_stop / dt)

    # Storage for results
    time = np.linspace(0, t_stop, num_steps)
    results = np.zeros((num_steps, num_nodes))

    # Initial condition (all zeros unless otherwise specified)
    v_prev = np.zeros(num_nodes)

    for step in range(num_steps):

        t = step * dt

        # Replace time-dependent sources with DC equivalents
        comp_t = evaluate_all_time_sources(components, t)

        Y, sources = generate_stamps_transient(comp_t, node_map, num_nodes, v_prev, dt, method)

        # Solve system
        v = solve_sparse(Y, sources)

        # Store results
        results[step, :] = v

        # Update state for next timestep
        v_prev = v.copy()

    return time, results

if __name__ == '__main__':
    file = r"testfiles/pulse_rc_circuit.txt"
    components = parse_netlist(file)
    dt = 0.1e-6
    t_stop = 40e-6
    time, results = transient_analysis_loop(components, t_stop, dt, "BE")
    plt.plot(time, results[:, 1])
    plt.show()