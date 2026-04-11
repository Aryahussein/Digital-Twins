"""
Noise Analysis Engine
=====================
Computes the output-referred noise spectral density S_v(f) in V²/Hz
for a circuit across a frequency range.

Theory
------
Each noise source k is modelled as an uncorrelated current source i_nk(t)
injected between two nodes, with power spectral density S_k(f) A²/Hz.

The noise contribution at the output node from source k is:

    S_out_k(f) = |H_k(f)|² * S_k(f)

where H_k(f) is the transfer function from the current injection point
of source k to the output voltage node.

Since all noise sources are uncorrelated, total output noise is:

    S_v_total(f) = Σ_k S_out_k(f)

Adjoint method
--------------
Computing H_k(f) for every source k separately would require N AC solves
per frequency (one per noise source). The adjoint method reduces this to
ONE backward solve per frequency.

We solve J^T * ψ = e_out where e_out is the unit vector selecting the
output node. Then:

    H_k(f) = ψ^T * e_k = ψ[p] - ψ[n]

where p, n are the positive and negative nodes of source k.

This gives all N transfer functions from a single adjoint solve — the
same trick used for sensitivity analysis.

Output
------
NoiseResult contains:
    frequencies  : array of Hz values
    S_v          : total output noise PSD in V²/Hz
    S_v_sqrt     : V/√Hz (input referred form is also common)
    contributions: dict {source_label: S_v_k array} per source
"""

import numpy as np


class NoiseResult:
    """Container for noise analysis results."""

    def __init__(self, frequencies, S_v, contributions):
        self.frequencies   = frequencies                    # Hz
        self.S_v           = S_v                           # V²/Hz total
        self.S_v_sqrt      = np.sqrt(np.maximum(S_v, 0))  # V/√Hz
        self.contributions = contributions                  # {label: array}

    def print_summary(self, at_freq=None):
        """Print noise summary at a specific frequency (or the geometric mean)."""
        if at_freq is None:
            at_freq = np.exp(np.mean(np.log(self.frequencies)))

        idx = np.argmin(np.abs(self.frequencies - at_freq))
        f0  = self.frequencies[idx]

        print(f"\n=== Noise Analysis Summary at f = {f0:.3e} Hz ===")
        print(f"  Total output noise: {self.S_v[idx]:.4e} V²/Hz")
        print(f"                      {self.S_v_sqrt[idx]:.4e} V/√Hz")
        print(f"\n  {'Source':<30} {'S_v (V²/Hz)':>15} {'%':>8}")
        total = self.S_v[idx]
        for label, S_arr in sorted(self.contributions.items(),
                                   key=lambda x: -x[1][idx]):
            S_k  = S_arr[idx]
            pct  = 100.0 * S_k / (total + 1e-300)
            print(f"  {label:<30} {S_k:>15.4e} {pct:>7.1f}%")


class NoiseEngine:
    """Computes AC noise spectral density for a circuit."""

    def __init__(self, circuit, output_node):
        """
        Parameters
        ----------
        circuit     : Circuit object (fully initialised)
        output_node : node name/int where noise voltage is observed
        """
        self.circuit     = circuit
        self.output_node = output_node

        # Resolve output index
        self._out_idx = (
            circuit.node_map.get(output_node)
            or circuit.node_map.get(int(output_node))
            or circuit.node_map.get(str(output_node))
        )
        if self._out_idx is None:
            raise ValueError(f"Output node '{output_node}' not in circuit.")

    def _solve_adjoint(self, lu):
        """One backward (adjoint) solve: J^T * ψ = e_out."""
        rhs = np.zeros(self.circuit.total_dim, dtype=complex)
        rhs[self._out_idx] = 1.0
        return lu.solve(rhs, trans='T')

    def run(self, Y_base, v_dc, ac_engine_result):
        """
        Compute noise PSD across all frequencies.

        Parameters
        ----------
        Y_base           : lil_matrix — static base admittance matrix
        v_dc             : DC operating point vector (for bias-dependent noise)
        ac_engine_result : tuple (frequencies, VI_ac, list_of_lus) from ACEngine.run()

        Returns
        -------
        NoiseResult
        """
        frequencies, VI_ac, list_of_lus = ac_engine_result

        n_freq        = len(frequencies)
        S_total       = np.zeros(n_freq)
        contributions = {}   # {label: np.zeros(n_freq)}

        print(f"\n--- Starting Noise Analysis ({n_freq} frequency points) ---")

        for fi, (f, lu, VI_f) in enumerate(
                zip(frequencies, list_of_lus, VI_ac)):

            if fi % max(1, n_freq // 10) == 0:
                print(f"  f = {f:.3e} Hz  ({fi+1}/{n_freq})")

            w = 2.0 * np.pi * f

            # One adjoint solve for this frequency
            psi = self._solve_adjoint(lu)

            # Collect all noise sources and their contributions.
            # IMPORTANT: noise source PSD depends on the DC operating point
            # (bias-dependent Id, gm) — NOT on the AC phasor VI_f.
            # Use v_dc for noise evaluation so that shot/channel noise
            # uses the correct DC bias current and transconductance.
            for comp in self.circuit.components:
                ns_list = comp.get_noise_sources(v_dc, w)
                for ns in ns_list:
                    p_idx, n_idx = ns['nodes']
                    S_k          = float(ns['S'])
                    label        = ns['label']

                    # Transfer function from this current source to output
                    # H_k = ψ[p] - ψ[n]  (sign convention: current from p→n)
                    psi_p = psi[p_idx].real if p_idx is not None else 0.0
                    psi_n = psi[n_idx].real if n_idx is not None else 0.0
                    H_k   = complex(psi[p_idx] if p_idx is not None else 0,
                                    0) - complex(
                                    psi[n_idx] if n_idx is not None else 0, 0)

                    S_out_k = (abs(H_k) ** 2) * S_k

                    S_total[fi]                                    += S_out_k
                    contributions.setdefault(label, np.zeros(n_freq))
                    contributions[label][fi]                       += S_out_k

        print(f"--- Noise Analysis Done ---")
        return NoiseResult(frequencies, S_total, contributions)
