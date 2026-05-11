"""AC Analysis Engine Module.

This module performs small-signal frequency domain simulations. It calculates 
the DC operating point to linearize nonlinear components (extracting small-signal 
parameters like gm and gds), and then sweeps the angular frequency to calculate 
complex phasors at every node.
"""

import numpy as np
from engines.solver import solve_linear_circuit
from engines.dc_engine import DCEngine

class ACEngine:
    """Evaluates the small-signal AC response of the circuit.

    Attributes:
        circuit (Circuit): The main circuit orchestrator.
        is_nonlinear (bool): Flag indicating if the circuit contains nonlinear 
            components that require small-signal linearization.
    """

    def __init__(self, circuit):
        """Initializes the AC Engine.

        Args:
            circuit (Circuit): The fully populated circuit orchestrator.
        """
        self.circuit = circuit
        self.is_nonlinear = circuit.is_nonlinear

    def compute(self, start_freq, stop_freq, points, sweep_type="DEC", keep_lus=False):
        """Executes the AC frequency sweep using a Linearize-Once optimization.

        Args:
            start_freq (float): Starting frequency in Hertz.
            stop_freq (float): Stopping frequency in Hertz.
            points (int or list): Number of frequency points to simulate. If 
                `sweep_type` is 'LIST', this must be an iterable of specific frequencies.
            sweep_type (str, optional): The scale of the sweep axis. Supported 
                options are 'DEC' (Decade), 'OCT' (Octave), 'LIN' (Linear), and 
                'LIST' (Custom array). Defaults to 'DEC'.
            keep_lus (bool, optional): If True, stores the LU factorization for each 
                frequency step (required for Adjoint sensitivity). Defaults to False.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list]: The frequency axis, the 
            2D complex voltage data array, and the cached LU objects.
        """
        sweep_type_upper = sweep_type.upper()

        # 1. Generate the Frequency Axis
        if sweep_type_upper == "LIN":
            freq_axis = np.linspace(start_freq, stop_freq, points)
        elif sweep_type_upper == "OCT":
            num_octaves = np.log2(stop_freq / start_freq)
            total_points = int(points * num_octaves)
            freq_axis = np.logspace(
                np.log2(start_freq), np.log2(stop_freq), 
                total_points, base=2.0
            )
        elif sweep_type_upper == "LIST":
            freq_axis = np.array(points)  
        else:  # DEC is the default
            freq_axis = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)

        print(f"\n--- Starting AC Sweep ({len(freq_axis)} points, {sweep_type_upper}) ---")

        VIs, list_of_lus = [], []

        # 2. Calculate the fundamental DC Operating Point (V_k)
        dc_engine = DCEngine(self.circuit)
        _, v_dc = dc_engine.compute_dc_bias(print_stuff=False)

        # =====================================================================
        # THE LINEARIZE-ONCE OPTIMIZATION
        # Ask the orchestrator to build the matrix at DC (w=0) using the bias 
        # point. This locks the small-signal gm/gds parameters into the matrix.
        # =====================================================================
        Y_lin_base, _ = self.circuit.build_system(domain="static", v_k=v_dc)
        
        # Cast the real DC matrix to complex memory ONCE so it can accept AC stamps
        Y_small_signal_base = Y_lin_base.astype(complex)

        # 3. Frequency Sweep Loop
        for i, f in enumerate(freq_axis):
            w = 2.0 * np.pi * f
            
            if i % max(1, len(freq_axis)//10) == 0: 
                print(f"Solving AC frequency {f:.2e} Hz")

            # The orchestrator uses the linearized base matrix and simply adds 
            # the fast, frequency-dependent stamps (jwC, 1/jwL) on top of it!
            Y_ac, J_ac = self.circuit.build_system(
                domain="frequency", w=w, v_k=v_dc, base_matrix=Y_small_signal_base
            )

            # AC is fundamentally linear, bypassing Newton-Raphson entirely
            lu_ac, v_ac = solve_linear_circuit(Y_ac.tocsc(), J_ac)

            VIs.append(v_ac)
            if keep_lus: 
                list_of_lus.append(lu_ac)

        print("--- AC Sweep Done ---")
                
        return freq_axis, np.array(VIs), list_of_lus
