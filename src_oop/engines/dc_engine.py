"""DC Analysis Engine Module.

This module is responsible for solving the static DC Operating Point (bias point)
of the circuit. The operating point is required before starting small-signal AC
analysis or providing the initial t=0 state for Transient integration.
"""

import numpy as np
from engines.solver import solve_linear_circuit, NonlinearSolver
from utils.utils import TemporaryCircuitState

class DCEngine:
    """Compiles circuit topology and calculates the steady-state DC bias.

    Attributes:
        circuit (Circuit): The main circuit orchestrator.
        is_nonlinear (bool): Flag indicating the presence of nonlinear devices
            requiring Newton-Raphson iteration.
    """

    def __init__(self, circuit):
        """Initializes the DC Engine.

        Args:
            circuit (Circuit): The fully populated circuit orchestrator.
        """
        self.circuit = circuit
        self.is_nonlinear = circuit.is_nonlinear

    def compute_dc_bias(self, v_ini=None, print_stuff=True):
        """Calculates the DC Operating Point of the circuit.

        Treats capacitors as open circuits and inductors as short circuits.
        
        Args:
            v_ini (numpy.ndarray, optional): An initial guess vector to speed 
                up Newton-Raphson convergence. Defaults to None (0V).
            print_stuff (bool, optional): Toggles console logging. Defaults to True.

        Returns:
            tuple[scipy.sparse.linalg.SuperLU, numpy.ndarray]: The cached LU 
            factorization and the solved node voltage array.
        """
        initial_guess = v_ini if v_ini is not None else np.zeros(self.circuit.total_dim)

        if self.is_nonlinear:
            solver = NonlinearSolver(self.circuit, print_stuff=print_stuff)
            return solver.solve(v_ini=initial_guess, domain="static")
            
        else:
            # Linear circuits converge instantly without iteration
            Y_dc, J_dc = self.circuit.build_system(domain="static")
            return solve_linear_circuit(Y_dc.tocsc(), J_dc)

    def compute_dc_sweep(self, source_name, start, stop, step, keep_lus=False):
        """Executes a large-signal DC sweep (.DC analysis).

        Args:
            source_name (str): The netlist name of the independent source to sweep.
            start (float): The starting value of the sweep.
            stop (float): The stopping value of the sweep.
            step (float): The increment step size.
            keep_lus (bool, optional): If True, stores the LU factorization for 
                each step (useful for adjoints). Defaults to False.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list]: The sweep axis, the 2D 
            voltage data array, and the list of cached LU objects.
        """
        sweep_axis = np.arange(start, stop + (step / 10.0), step)
        VIs, list_of_lus = [], []
        
        print(f"\n--- Starting DC Sweep ({len(sweep_axis)} points) ---")

        current_guess = np.zeros(self.circuit.total_dim)
        # param_name = f"{source_name}_value"
        param_name = f"{source_name}_value"
        if param_name not in self.circuit.param_to_component_map:
            param_name = source_name


        with TemporaryCircuitState(self.circuit, self.circuit.param_to_component_map, {param_name: start}):
            
            for idx, val in enumerate(sweep_axis):
                if idx % max(1, len(sweep_axis)//10) == 0: 
                    print(f"Solving DC sweep point: {val:.3f}")

                # Safely update the source value
                comp = self.circuit.param_to_component_map[param_name]
                comp.set_nominal_value(param_name, val)
                
                lu_dc, VI_dc = self.compute_dc_bias(v_ini=current_guess, print_stuff=False)
                
                current_guess = VI_dc.copy() 
                VIs.append(VI_dc)
                if keep_lus: 
                    list_of_lus.append(lu_dc)

        return sweep_axis, np.array(VIs), list_of_lus
