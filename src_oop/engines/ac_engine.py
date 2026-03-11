"""
AC Analysis Engine Module.

This module handles small-signal frequency domain simulations. It takes a resolved
DC operating point, linearizes all nonlinear components around that bias point,
and sweeps the frequency to calculate the complex phasors (magnitude and phase)
at every node.
"""

import numpy as np
from engines.solver import solve_linear_circuit

class ACEngine:
    """Evaluates the small-signal AC response of the circuit.

    Attributes:
        circuit (Circuit): The main circuit object containing components and node mappings.
        is_nonlinear (bool): Flag indicating if the circuit contains nonlinear components
            that require small-signal linearization (e.g., Diodes, MOSFETs).
    """

    def __init__(self, circuit, is_nonlinear):
        """Initializes the AC Engine.

        Args:
            circuit (Circuit): The fully populated circuit object.
            is_nonlinear (bool): Boolean flag denoting presence of nonlinear devices.
        """
        self.circuit = circuit
        self.is_nonlinear = is_nonlinear
        
    def _solve_single_point(self, w, Y_small_signal_base):
        """Solves the complex AC circuit for a single angular frequency.

        Args:
            w (float): The angular frequency ($w = 2 \pi f$) in rad/s.
            Y_small_signal_base (scipy.sparse.lil_matrix): The base admittance matrix
                already containing static resistors and linearized nonlinear conductances.

        Returns:
            tuple: (lu_factorization, VI_complex_array) representing the solved 
            state of the circuit at this exact frequency.
        """
        # 1. Start with the pre-linearized matrix (Massive Speed Optimization!)
        Y_ac = Y_small_signal_base.copy()
        
        # 2. AC sources must be completely fresh (DC sources are killed in AC analysis)
        sources_ac = np.zeros(self.circuit.total_dim, dtype=complex)

        # 3. Only stamp the frequency-dependent dynamic terms (C, L) and AC phasors (V, I)
        for comp in self.circuit.components:
            comp.stamp_ac(Y_ac, sources_ac, w)
                
        # 4. Convert to CSC right before solving to prevent matrix corruption
        return solve_linear_circuit(Y_ac.tocsc(), sources_ac)

    def run(self, Y_base_lil, VI_dc, start_freq, stop_freq, points, sweep_type="DEC", keep_lus=False):
        """Executes the AC frequency sweep.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The static base admittance matrix.
            VI_dc (np.ndarray): The resolved DC operating point vector.
            start_freq (float): Starting frequency in Hz.
            stop_freq (float): Stopping frequency in Hz.
            points (int): Number of frequency points to simulate.
            sweep_type (str, optional): The spacing of the sweep ('DEC' for logarithmic, 
                'LIN' for linear). Defaults to 'DEC'.
            keep_lus (bool, optional): If True, stores the LU factorization for each 
                frequency step (required for Adjoint sensitivity). Defaults to False.

        Returns:
            tuple: (frequencies, VIs, list_of_lus) containing the frequency axis, 
            the complex 2D results array, and the cached matrix factorizations.
        """
        # Extension: Support both Linear and Decade (Logarithmic) sweeps
        if sweep_type.upper() == "LIN":
            frequencies = np.linspace(start_freq, stop_freq, points)
        else:
            frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)
            
        VIs, list_of_lus = [], []
        
        # =========================================================================
        # THE LINEARIZE-ONCE OPTIMIZATION
        # Bake the small-signal parameters (gm, gds) into a base AC matrix ONCE.
        # =========================================================================
        Y_small_signal_base = Y_base_lil.copy().astype(complex)
        
        if self.is_nonlinear:
            dummy_dc_sources = np.zeros(self.circuit.total_dim, dtype=complex)
            for comp in self.circuit.components:
                # Stamp the small-signal conductances evaluated at the DC bias point
                comp.stamp_nonlinear(Y_small_signal_base, dummy_dc_sources, p_V_guess=VI_dc, V_guess=VI_dc)
        
        print(f"\n--- Starting AC Sweep ({points} points, {sweep_type}) ---")
        
        for step, f in enumerate(frequencies):
            if step % max(1, len(frequencies)//10) == 0: 
                print(f"Solving AC frequency {f:.2e} Hz")
                
            w = 2 * np.pi * f
            
            # Pass the PRE-LINEARIZED matrix into the solver
            lu_ac, VI_ac = self._solve_single_point(w, Y_small_signal_base)
            
            VIs.append(VI_ac)
            if keep_lus: 
                list_of_lus.append(lu_ac)
                
        return frequencies, np.array(VIs), list_of_lus
