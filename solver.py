from assembleYmatrix import generate_stamps, update_nonlinear_stamps
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
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

def solve_adjoint(lu, target, var_map, total_dim):
    """
    target can be a node number (e.g. 2) or a name (e.g. 'V1')
    """
    d = np.zeros(total_dim)
    idx = var_map[target]  # Works for both nodes and branches!
    d[idx] = 1.0
    return lu.solve(d, trans='T')

def solve_nonlinear_circuit(Y_ori, sources_ori, components, node_map, total_dim, V_ini, max_iter=100, tol=1e-6, num_steps=10):
    # Source Ramping: k goes from 0.1 to 1.0
    ramp = np.linspace(1.0/num_steps, 1.0, num_steps)

    V_k = V_ini.copy()
    lu = np.empty((total_dim, total_dim))
    for k in ramp:
        print(f"\n--- Ramping Source: {k*100:.1f}% ---")
        sources_k = sources_ori.copy() * k
        Y_k = Y_ori.copy()
        
        for i in range(max_iter):
            # 1. Generate linearized Y and I for the current guess V_k
            # Sources in I_rhs should be multiplied by k_factor inside generate_stamps
            prev_V_k = V_k.copy()
            
            Y_iter, sources_iter = update_nonlinear_stamps(Y_k, sources_k, components, node_map, prev_V_k, V_k)
            
            # 2. Solve the linear system
            lu, V_new = solve_linear_circuit(Y_iter, sources_iter)
            
            # 3. Check for convergence
            # Delta V is the difference between the new result and the previous guess
            delta_v = np.abs(V_new - V_k)
            max_error = np.max(delta_v)
            
            V_k = V_new # Update guess for next iteration
            
            if max_error < tol:
                print(f"Converged in {i+1} iterations.")
                break
        else:
            raise RuntimeError(f"Newton-Raphson failed to converge at {k*100}% ramp.")
            
    return lu, V_k # Return final solution and the last factorized matrix
