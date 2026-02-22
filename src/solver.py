from scipy.sparse.linalg import spsolve, splu
from assembleYmatrix import stamp_nonlinear_components
import numpy as np
from constants import *

def solve_sparse(G, I):
    return spsolve(G, I) 

def solve_LU(G):
    return splu(G)

def get_node_and_branch_currents(lu, sources):
    VI = lu.solve(sources)
    return VI

def solve_linear_circuit(Y, sources):
    # solve! get LU, node voltages and branch currents of original circuit
    lu = solve_LU(Y)
    VI = get_node_and_branch_currents(lu, sources)
    return lu, VI

def solve_adjoint(lu, target, node_map):
    """
    target can be a node number (e.g. 2) or a name (e.g. 'V1')
    """
    d = np.zeros(len(node_map))
    idx = node_map[target]  # Works for both nodes and branches!
    d[idx] = 1.0
    return lu.solve(d, trans='T')

def solve_nonlinear_circuit(Y_base, sources_base, components, node_map, V_ini, max_iter=100, tol=1e-6, num_steps=10):
    """Newton-Raphson solver with source ramping for nonlinear circuits."""
    # Source Ramping: k goes from 1/num_steps to 1.0
    ramp = np.linspace(1.0/num_steps, 1.0, num_steps)

    V_k = V_ini.copy()
    prev_V_k = V_ini.copy()
    lu = None  # Will be set by solve_linear_circuit in the loop

    for k in ramp:
        print(f"\n--- Ramping Source: {k*100:.1f}% ---")
        sources_k = sources_base.copy() * k
        Y_k = Y_base.copy()
        
        for i in range(max_iter):
            Y_iter, sources_iter = Y_k.copy(), sources_k.copy()
            # 1. Generate linearized Y and I for the current guess V_k
            Y_iter, sources_iter = stamp_nonlinear_components(Y_iter, sources_iter, components, node_map, prev_V_k, V_k)
            
            # 2. Solve the linear system
            lu, V_new = solve_linear_circuit(Y_iter, sources_iter)
            
            # 3. Check for convergence
            # Delta V is the difference between the new result and the previous guess
            delta_v = np.abs(V_new - V_k)
            max_error = np.max(delta_v)
            
            prev_V_k = V_k.copy()
            V_k = V_new # Update guess for next iteration
            print(f"Iteration: {i}, error = {max_error}")
            if max_error < tol:
                print(f"Converged in {i+1} iterations.")
                break
        else:
            raise RuntimeError(f"Newton-Raphson failed to converge at {k*100}% ramp.")
            
    return lu, V_k # Return final solution and the last factorized matrix

