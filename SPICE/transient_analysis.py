import numpy as np
from assembleYmatrix import generate_transient_basis, stamp_nonlinear_components, NONLINEAR_DISPATCH
from sources import evaluate_all_time_sources
from solver import solve_sparse
from node_index import build_node_index
from txt2dictionary import parse_netlist
import matplotlib.pyplot as plt

def transient_analysis_loop(components, t_stop, dt, method="BE"):
    """
    Performs transient analysis with support for both linear (RLC) 
    and non-linear (Diode) components using Newton-Raphson at each time step.
    """
    node_map, total_dim = build_node_index(components)
    num_steps = int(t_stop / dt)

    # Optimization: Check if we actually have non-linear parts
    # If not, we can skip the inner iteration loop
    has_nonlinear = any(name[0] in NONLINEAR_DISPATCH for name in components)

    # Arrays to store simulation results
    time = np.linspace(0, t_stop, num_steps)
    results = np.zeros((num_steps, total_dim))

    # Initial Conditions
    # Ideally, run a DC Operating Point (.op) here to get valid initial start
    # For now, we assume zero initial state
    v_prev = np.zeros(total_dim)
    sources_prev = np.zeros(total_dim)

    # Newton-Raphson Configuration
    MAX_ITER = 50
    TOLERANCE = 1e-6

    print(f"Starting Simulation: {t_stop*1000:.2f}ms, dt={dt*1e6:.2f}us")
    print(f"Non-Linear Mode: {'Enabled' if has_nonlinear else 'Disabled (Linear Only)'}")

    for step in range(num_steps):
        t = step * dt

        # 1. Update Time-Varying Sources (Pulse, Sin, etc.)
        comp_t = evaluate_all_time_sources(components, t)

        # 2. Build Linear Basis Matrix
        # These parts (R, L_eq, C_eq) are constant for this time step
        Y_base, sources_base = generate_transient_basis(comp_t, node_map, total_dim, dt, v_prev, sources_prev)

        # 3. Newton-Raphson Loop
        # Initial guess is the voltage from the previous timestep (Time Projection)
        v_guess = v_prev.copy()
        
        # If the circuit is linear, we only need 1 pass.
        iterations = MAX_ITER if has_nonlinear else 1
        
        for i in range(iterations):
            # Work on copies so we don't pollute the base matrix with bad guesses
            Y_iter = Y_base.copy()
            sources_iter = sources_base.copy()

            # Stamp Non-Linear parts at the current guess
            if has_nonlinear:
                stamp_nonlinear_components(Y_iter, sources_iter, comp_t, node_map, v_guess)

            # Solve the system
            try:
                v_new = solve_sparse(Y_iter.tocsc(), sources_iter)
            except RuntimeError:
                print(f"Matrix Singular at t={t}")
                break

            # Check Convergence
            if not has_nonlinear:
                v_guess = v_new
                break
                
            delta = np.max(np.abs(v_new - v_guess))
            if delta < TOLERANCE:
                v_guess = v_new
                # Optional: print(f"Step {step} converged in {i} iterations")
                break
            
            v_guess = v_new # Update guess for next iter
        
        # 4. Advance Time
        results[step, :] = v_guess
        v_prev = v_guess.copy()
        sources_prev = sources_base.copy()

    return time, results

if __name__ == '__main__':
    # Usage Example
    # Ensure you have a netlist file available
    file_path = "testfiles/buck_circuit.txt" 
    
    try:
        components, analyses = parse_netlist(file_path)
        
        # Extract simulation parameters from .tran directive if available
        # Otherwise default to:
        t_stop = 2e-3
        dt = 1e-6
        
        if ".tran" in analyses:
            t_stop = analyses[".tran"]["stop_time"]
            dt = analyses[".tran"]["max_timestep"]

        time, res = transient_analysis_loop(components, t_stop, dt)
        
        # Plotting results (Assuming Node 2 is Output)
        if res.shape[1] > 2:
            plt.figure(figsize=(10, 5))
            plt.plot(time * 1000, res[:, 2], label="Output Voltage")
            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (V)")
            plt.title("Transient Response")
            plt.grid(True)
            plt.legend()
            plt.show()
            
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")