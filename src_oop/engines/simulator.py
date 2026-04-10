"""
Master Simulator Module.

This module provides the central `Simulator` class, which orchestrates the execution
of various circuit analyses. It initializes the base circuit topology and dynamically 
routes the data to the correct numerical engine (DC, AC, Transient) based on the 
user's requested simulation commands.
"""

from engines.dc_engine import DCEngine
from engines.transient_engine import TransientEngine
from engines.ac_engine import ACEngine
from engines.adjoint_engine import AdjointEngine
from core.results import SimulationResult
import numpy as np

class Simulator:
    """The master orchestrator for circuit simulation.

    Reads the requested analysis types from the parsed netlist and triggers 
    the appropriate simulation engines, ultimately packaging the results into 
    a secure `SimulationResult` vault.

    Attributes:
        circuit (Circuit): The fully initialized, polymorphic circuit object.
        analyses (dict): A dictionary of requested analyses parsed from the netlist 
            (e.g., {'.TRAN': {'stop': 1e-3, 'step': 1e-6}}).
        output_nodes (list): Target nodes for Adjoint sensitivity calculations.
        is_nonlinear (bool): Automatically detected flag indicating if Newton-Raphson 
            solvers are required.
        is_complex (bool): Automatically detected flag indicating if complex matrix 
            dtypes are required for AC analysis.
    """

    def __init__(self, circuit, analyses, output_nodes=None):
        """Initializes the Simulator environment.

        Args:
            circuit (Circuit): The constructed circuit object.
            analyses (dict): The dictionary of simulation commands.
            output_nodes (list, optional): Specific nodes to track for sensitivity. 
                Defaults to tracking all voltage nodes.
        """
        self.circuit = circuit
        self.analyses = analyses
        self.output_nodes = output_nodes if output_nodes else list(circuit.node_map.keys())
        
        # Check flags by peeking into the object types
        self.is_nonlinear = any(comp.IS_NONLINEAR for comp in circuit.components)
        self.is_complex = ".AC" in analyses

    def execute_analysis(self, sensitivity=False, keep_lus=False):
        """Routes and executes the requested simulation.

        This method builds the static matrix topology once, then checks the 
        `analyses` dictionary to determine which engine to invoke. 

        Args:
            sensitivity (bool, optional): If True, triggers the backward Adjoint 
                pass after the primary simulation finishes. Defaults to False.
            keep_lus (bool, optional): If True, caches the scipy LU factorizations 
                in memory. Automatically enabled if sensitivity is True. Defaults to False.

        Returns:
            SimulationResult: A secure data vault containing the simulation outputs.

        Raises:
            ValueError: If no valid analysis commands (.TRAN, .AC, .DC, .OP) are found.
        """
        # 1. Base Topology Setup
        dc_engine = DCEngine(self.circuit, self.is_complex, self.is_nonlinear)
        Y_base = dc_engine.build_base_matrices()

        # ==========================================
        # TRANSIENT ROUTE
        # ==========================================
        if ".TRAN" in self.analyses:
            print("\nStarting Transient Analysis...")
            t_stop = self.analyses[".TRAN"]["stop"]
            dt = self.analyses[".TRAN"]["step"]
            
            # Get t=0 starting bias (Removed sources_base)
            _, v_initial = dc_engine.compute_dc_bias(Y_base)
            
            # Run Forward Engine (Removed sources_base)
            tran_engine = TransientEngine(self.circuit, self.is_nonlinear)
            method = self.analyses.get("OPTIONS", {}).get("method", "TR")
            time, VI, lus = tran_engine.run(
                Y_base, v_initial, t_stop, dt, keep_lus=(keep_lus or sensitivity), method = method
            )
            
            result = SimulationResult(".TRAN", time, VI, self.circuit.node_map, dt=dt)
            result.list_of_lus = lus
            
            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                adjoint_results = {}

                global_sens = adj_engine.compute_transient(time, VI, lus, dt)
                adjoint_results.update(global_sens) # Adds "Time_Series" and "Integrated_Transient"

                # local_sens = adj_engine.compute_continuous_local_adjoint(time, VI, lus, dt)
                # adjoint_results["Continuous_Local_DC"] = local_sens
                cube = adj_engine.compute_continuous_local_adjoint(time, VI, lus, dt)
                adjoint_results["Sensitivity_Cube"] = cube

                result.sensitivities = adjoint_results
                
            return result

        # ==========================================
        # AC ROUTE
        # ==========================================
        elif ".AC" in self.analyses:
            print("\nStarting AC Analysis...")
            start = self.analyses[".AC"]["start"]
            stop = self.analyses[".AC"]["stop"]
            pts = self.analyses[".AC"]["num_points"]
            sweep_type = self.analyses[".AC"].get("sweep_type", "DEC")
            
            # 1. Get the DC linearization point
            _, v_dc = dc_engine.compute_dc_bias(Y_base)
            
            ac_engine = ACEngine(self.circuit, self.is_nonlinear)
            freq, VI, lus = ac_engine.run(
                Y_base, v_dc, start, stop, pts, sweep_type=sweep_type, keep_lus=(keep_lus or sensitivity)
            )
            
            result = SimulationResult(".AC", freq, VI, self.circuit.node_map)
            result.list_of_lus = lus

            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(VI, lus, freq_array=freq)
                
            return result

        # ==========================================
        # DC SWEEP ROUTE
        # ==========================================
        elif ".DC" in self.analyses:
            source_name = self.analyses[".DC"]["source"]
            start = self.analyses[".DC"]["start"]
            stop = self.analyses[".DC"]["stop"]
            step = self.analyses[".DC"]["step"]
            
            sweep_axis, VI, lus = dc_engine.compute_dc_sweep(
                Y_base, source_name, start, stop, step, keep_lus=(keep_lus or sensitivity)
            )
            
            result = SimulationResult(".DC", sweep_axis, VI, self.circuit.node_map)
            result.list_of_lus = lus

            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(VI, lus)
                
            return result
            
        # ==========================================
        # OPERATING POINT (.OP) ROUTE
        # ==========================================
        elif ".OP" in self.analyses:
            print("\nStarting DC Operating Point Analysis...")
            
            lu_dc, v_dc = dc_engine.compute_dc_bias(Y_base)
            
            # Store single point in Data Vault
            result = SimulationResult(".OP", np.array([0.0]), v_dc, self.circuit.node_map)
            result.list_of_lus = [lu_dc]

            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                # Wrap the 1D v_dc vector into a 2D array [v_dc] so the sweep engine can iterate it
                result.sensitivities = adj_engine.compute_sweep(np.array([v_dc]), [lu_dc])
            
            return result

        else:
            raise ValueError("No recognized analysis (.TRAN, .AC, .DC, .OP) found in the parsed netlist commands.")
