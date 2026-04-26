"""
Large Change Sensitivity specifically adding a resistance between two nodes:

Computes the exact effect of adding/modifying a resistance between 
two nodes using the Sherman-Morrison / Kron's formula.

    Given a fault resistance R added between nodes k and l, the new solution is:
    
        v_hat = v - (v_oc / (R + R_TH)) * Y_inv_xi 
    where:
        v_oc     = xi_kl^T * v            (open circuit voltage between k and l)
        R_TH     = xi_kl^T * Y^-1 * xi_kl (Thevenin resistance seen from k,l)
        Y_inv_xi = Y^-1 * xi_kl           (one LU forward solve)
        xi_kl    = connection vector       (+1 at node k, -1 at node l)
    
No new LU factorizations are needed — only the original LU factors 
and the forward solution are required.

"""

import numpy as np


def build_xi(n, idx_k, idx_l):
    """
    Build the connection vector xi_kl.
    
    xi_kl has +1 at node k, -1 at node l, and 0 everywhere else.
    This defines the orientation of a branch from node k to node l.
    """
    xi_kl = np.zeros(n)
    if idx_k is not None:
        xi_kl[idx_k] = 1.0
    if idx_l is not None:
        xi_kl[idx_l] = -1.0
    return xi_kl


def compute_large_change(lu, xi_kl, v, R, output_idx):
    """
    Compute the exact change in output voltage when a resistance R 
    is added between two nodes:
    
        delta_v = -(v_oc / (R + R_TH)) * Y_inv_xi[output_idx]
    
    where:
        v_oc     = xi_kl^T * v             (open circuit voltage between node k and l)
        R_TH     = xi_kl^T * Y^-1 * xi_kl  (Thevenin resistance between node k and l)
        Y_inv_xi = Y^-1 * xi_kl            (one LU forward solve)
        i_R      = v_oc / (R + R_TH)       (loop current)

    Args:
        lu: The LU factorization of the Y matrix at this time step.
        xi_kl (np.ndarray): Connection vector for the fault branch.
        v (np.ndarray): The nominal solution vector at this time step.
        R (float): The value of the added/fault resistance.
        output_idx (int): The matrix index of the output node to observe.
    
    Returns:
        float: The change in output voltage delta_v_out.
    """
    v_oc = xi_kl @ v
    Y_inv_xi = lu.solve(xi_kl)
    R_TH = xi_kl @ Y_inv_xi

    denominator = R + R_TH
    if abs(denominator) < 1e-15: #safety check
        return np.inf

    # Current through the fault resistance
    i_R = v_oc / denominator
    delta_v_out = -i_R * Y_inv_xi[output_idx]

    return delta_v_out
