from scipy.sparse.linalg import spsolve, splu
from assembleYmatrix import stamp_nonlinear_components
import numpy as np
from constants import *

def solve_sparse(G, I):
    return spsolve(G, I)

def solve_LU(G):
    return splu(G)

def get_node_and_branch_currents(lu, sources):
    return lu.solve(sources)

def solve_linear_circuit(Y, sources):
    lu = solve_LU(Y)
    VI = get_node_and_branch_currents(lu, sources)
    return lu, VI

def solve_adjoint(lu, target, node_map):
    d = np.zeros(len(node_map))
    idx = node_map[target]
    d[idx] = 1.0
    return lu.solve(d, trans='T')

def solve_nonlinear_circuit(Y_base, sources_base, components, node_map, V_ini,
                           max_iter=100, tol=1e-6, num_steps=10, print_stuff=True):
    ramp = np.linspace(1.0/num_steps, 1.0, num_steps)

    V_k = V_ini.copy()
    prev_V_k = V_ini.copy()
    lu = None

    for k in ramp:
        # Gain ramping for opamps
        for comp in components.values():
            if comp.get("type") == "O":
                comp["_gain_scale"] = k

        if print_stuff:
            print(f"\n--- Ramping Source: {k*100:.1f}% ---")

        sources_k = sources_base.copy() * k
        Y_k = Y_base.copy()

        for i in range(max_iter):
            Y_iter, sources_iter = Y_k.copy(), sources_k.copy()

            # Stamp nonlinear (and opamp gain-scaled) elements around current guess
            Y_iter, sources_iter = stamp_nonlinear_components(
                Y_iter, sources_iter, components, node_map, prev_V_k, V_k
            )

            lu, V_new = solve_linear_circuit(Y_iter, sources_iter)

            delta_v = np.abs(V_new - V_k)
            max_error = np.max(delta_v)

            prev_V_k = V_k.copy()

            alpha = 1.0  # set <1.0 for damping if needed
            V_k = V_k + alpha * (V_new - V_k)

            if print_stuff:
                print(f"Iteration: {i}, error = {max_error}")

            if max_error < tol:
                if print_stuff:
                    print(f"Converged in {i+1} iterations.")
                break
        else:
            raise RuntimeError(f"Newton-Raphson failed to converge at {k*100:.1f}% ramp.")

    # Reset opamp scaling
    for comp in components.values():
        if comp.get("type") == "O":
            comp["_gain_scale"] = 1.0

    return lu, V_k

