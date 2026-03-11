"""
DC Sweep Engine Module.

This module performs large-signal DC sweeps (.DC analysis). It iteratively 
changes the value of a specific component (usually a voltage or current source),
recalculates the nonlinear DC Operating Point, and compiles the results into 
a continuous sweep array (e.g., for plotting I-V curves).
"""

import numpy as np

class DCSweepEngine:
    """Evaluates the large-signal DC response over a range of parameter values.

    Attributes:
        circuit (Circuit): The main circuit object containing components.
        dc_engine (DCEngine): A reference to the initialized DC engine used to
            calculate the baseline operating points.
    """

    def __init__(self, circuit, dc_engine):
        """Initializes the DC Sweep Engine.

        Args:
            circuit (Circuit): The fully populated circuit object.
            dc_engine (DCEngine): The engine responsible for calculating the 
                steady-state bias points.
        """
        self.circuit = circuit
        self.dc_engine = dc_engine

    def run(self, Y_base_lil, sources_base, source_name, start, stop, step, keep_lus=False):
        """Executes the DC sweep analysis.

        Args:
            Y_base_lil (scipy.sparse.lil_matrix): The static base admittance matrix.
            sources_base (np.ndarray): The static base RHS current/voltage vector.
            source_name (str): The netlist name of the component to sweep (e.g., 'V1').
            start (float): The starting value of the sweep.
            stop (float): The stopping value of the sweep.
            step (float): The increment step size.
            keep_lus (bool, optional): If True, stores the LU factorization for each 
                sweep step. Defaults to False.

        Returns:
            tuple: (sweep_axis, VIs, list_of_lus) containing the 1D numpy array 
            of swept values, the 2D solution matrix, and cached LU factors.

        Raises:
            KeyError: If the specified `source_name` does not exist in the circuit.
        """
        # 1. Find the component we are sweeping using our fast O(1) dictionary lookup
        target_comp = self.circuit.get_component(source_name)

        # 2. Create the sweep axis 
        # Add a tiny amount to 'stop' to ensure floating point math includes the final point
        sweep_axis = np.arange(start, stop + (step / 10.0), step)
        VIs, list_of_lus = [], []
        
        print(f"\n--- Starting DC Sweep ({len(sweep_axis)} points) ---")

        # 3. Save the original value so we don't permanently break the circuit!
        original_value = target_comp.value

        # Start with a guess of 0V for the very first point
        current_guess = np.zeros(self.circuit.total_dim)

        # 4. Sweep!
        for idx, val in enumerate(sweep_axis):
            if idx % max(1, len(sweep_axis)//10) == 0: 
                print(f"Solving DC sweep point: {val:.3f}")

            # Temporarily overwrite the component's value
            target_comp.value = val
            
            # Re-run the DC operating point. 
            # We pass print_stuff=False to mute the NR solver spam, and we pass
            # current_guess forward to make convergence almost instantaneous!
            lu_dc, VI_dc = self.dc_engine.compute_dc_bias(
                Y_base_lil, 
                sources_base, 
                v_ini=current_guess, 
                print_stuff=False
            )
            
            # Save the new answer to be used as the guess for the next loop iteration
            current_guess = VI_dc.copy() 
            
            VIs.append(VI_dc)
            if keep_lus: 
                list_of_lus.append(lu_dc)

        # 5. Restore the component to its original state so subsequent analyses aren't affected
        target_comp.value = original_value

        return sweep_axis, np.array(VIs), list_of_lus
