"""
Moments Engine
==============
Computes the moments of the circuit transfer function H(s) = V_out(s)/V_in(s).

The transfer function is expanded as a Taylor series around s=0:

    H(s) = m_0 + m_1*s + m_2*s^2 + ... + m_{N-1}*s^{N-1}

The moments are computed recursively using the circuit MNA matrices:

    x_0 = G^{-1} * b                  (DC solution vector)
    x_k = -G^{-1} * C * x_{k-1}      (k-th moment vector)

    m_k = e_out^T * x_k               (scalar moment at output node)

where:
    G   = DC conductance matrix  (resistors + voltage source stamps)
    C   = capacitance matrix     (capacitors only — built separately)
    b   = DC excitation vector   (voltage/current source values)
    e_out = unit vector selecting the output node

Algorithm
---------
1. Build G matrix and b vector (same as DC operating point).
2. Build C matrix separately by stamping only capacitors.
3. LU-factorise G once.
4. Solve recursively: x_k = LU^{-1} * (-C * x_{k-1}).
5. Extract scalar moment at output node: m_k = x_k[output_idx].
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu


class MomentsEngine:
    """Computes transfer function moments for linear RC/RLC circuits."""

    def __init__(self, circuit):
        self.circuit = circuit

    def _build_G_and_b(self):
        """
        Build the DC conductance matrix G and excitation vector b.

        This is equivalent to what DCEngine does for a .OP analysis,
        except we keep them separate for the recursive moment computation.
        """
        dim  = self.circuit.total_dim
        G    = lil_matrix((dim, dim), dtype=float)
        b    = np.zeros(dim, dtype=float)

        # Stamp static (resistors, MNA branch connections)
        for comp in self.circuit.components:
            comp.stamp_mna_connection(G)
            comp.stamp_static(G)

        # Stamp source excitation into b.
        # Use DC value if nonzero, otherwise fall back to AC magnitude.
        # This ensures moments work for both DC-driven and AC-driven circuits.
        for comp in self.circuit.components:
            comp.stamp_dc(G, b)

        # If b is all zeros (e.g. source is defined as AC only), use AC phasors
        if np.all(b == 0):
            for comp in self.circuit.components:
                if hasattr(comp, 'ac_mag') and comp.ac_mag != 0:
                    # Stamp unit AC excitation as a real value
                    if hasattr(comp, 'branch_idx') and comp.branch_idx is not None:
                        b[comp.branch_idx] += float(comp.ac_mag)

        return G.tocsc(), b

    def _build_C(self):
        """
        Build the capacitance matrix C by stamping only capacitors.

        Each capacitor C between nodes i and j contributes:
            C[i,i] += C_val
            C[j,j] += C_val
            C[i,j] -= C_val
            C[j,i] -= C_val

        This is the structure that appears in the s*C term of the
        MNA equation: (G + s*C) * V = b
        """
        dim = self.circuit.total_dim
        C   = lil_matrix((dim, dim), dtype=float)

        for comp in self.circuit.components:
            if comp.type == 'C':
                i = comp.idx_1
                j = comp.idx_2
                val = comp.value
                if i is not None:
                    C[i, i] += val
                    if j is not None:
                        C[i, j] -= val
                        C[j, i] -= val
                if j is not None:
                    C[j, j] += val

            elif comp.type == 'L':
                # Inductor in s-domain: Z = s*L in the branch equation
                # The branch equation row gets -s*L on diagonal
                b_idx = getattr(comp, 'branch_idx', None)
                if b_idx is not None:
                    C[b_idx, b_idx] -= comp.value

        return C.tocsc()

    def compute(self, output_node, num_moments=10):
        """
        Compute the first num_moments moments of H(s) = V(output_node)(s) / V_in(s).

        Parameters
        ----------
        output_node : node name or int
            The node at which to evaluate the transfer function.
        num_moments : int
            Number of moments to compute (default 10).

        Returns
        -------
        moments : np.ndarray shape (num_moments,)
            m_k = k-th Taylor coefficient of H(s) around s=0.
        x_vecs  : list of np.ndarray
            Full solution vectors x_k for each moment (needed by AWE).
        """
        # Resolve output node index
        out_idx = self.circuit.node_map.get(output_node)
        if out_idx is None:
            try:
                out_idx = self.circuit.node_map.get(int(output_node))
            except (ValueError, TypeError):
                pass
        if out_idx is None:
            raise ValueError(f"Output node '{output_node}' not found.")

        # Build matrices
        G, b = self._build_G_and_b()
        C    = self._build_C()

        # LU factorise G once — reused for all moment solves
        try:
            lu = splu(G)
        except Exception as e:
            raise RuntimeError(f"G matrix singular — check circuit: {e}") from e

        # Recursive moment computation
        moments = np.zeros(num_moments, dtype=float)
        x_vecs  = []

        x_prev = lu.solve(b)                  # x_0 = G^{-1} * b
        moments[0] = x_prev[out_idx]
        x_vecs.append(x_prev.copy())

        for k in range(1, num_moments):
            rhs    = -(C @ x_prev)            # -C * x_{k-1}
            x_k    = lu.solve(rhs)            # G^{-1} * (-C * x_{k-1})
            moments[k] = x_k[out_idx]
            x_vecs.append(x_k.copy())
            x_prev = x_k

        return moments, x_vecs

    def frequency_response(self, output_node, frequencies, num_moments=10):
        """
        Evaluate the moment-based transfer function at given frequencies.

        H(j*omega) ≈ sum_{k=0}^{N-1} m_k * (j*omega)^k

        Parameters
        ----------
        frequencies : array-like
            Frequency values in Hz.
        num_moments : int
            Number of moments to use in the approximation.

        Returns
        -------
        H : np.ndarray (complex), same length as frequencies
        """
        moments, _ = self.compute(output_node, num_moments)
        freqs  = np.asarray(frequencies)
        omegas = 2 * np.pi * freqs
        H      = np.zeros(len(freqs), dtype=complex)

        for k, m_k in enumerate(moments):
            H += m_k * (1j * omegas) ** k

        return H
