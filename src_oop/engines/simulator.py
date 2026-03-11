"""
Master Simulator Module.

This module provides the central `Simulator` class, which orchestrates the execution
of various circuit analyses. It initializes the base circuit topology and dynamically 
routes the data to the correct numerical engine (DC, AC, Transient) based on the 
user's requested simulation commands.
"""

from engines.dc_engine import DCEngine
from engines.transient_engine import TransientEngine
from engines.dc_sweep_engine import DCSweepEngine
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
        ramp (int): Legacy source-stepping parameter for DC convergence.
        is_nonlinear (bool): Automatically detected flag indicating if Newton-Raphson 
            solvers are required.
        is_complex (bool): Automatically detected flag indicating if complex matrix 
            dtypes are required for AC analysis.
    """

    def __init__(self, circuit, analyses, output_nodes=None, ramp=1):
        """Initializes the Simulator environment.

        Args:
            circuit (Circuit): The constructed circuit object.
            analyses (dict): The dictionary of simulation commands.
            output_nodes (list, optional): Specific nodes to track for sensitivity. 
                Defaults to tracking all voltage nodes.
            ramp (int, optional): Source step parameter. Defaults to 1.
        """
        self.circuit = circuit
        self.analyses = analyses
        self.output_nodes = output_nodes if output_nodes else list(circuit.node_map.keys())
        self.ramp = ramp
        
        # Check flags by peeking into the object types
        self.is_nonlinear = any(comp.type in ['D', 'M'] for comp in circuit.components)
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
        dc_engine = DCEngine(self.circuit, self.is_complex, self.is_nonlinear, self.ramp)
        Y_base, sources_base = dc_engine.build_base_matrices()

        # ==========================================
        # TRANSIENT ROUTE
        # ==========================================
        if ".TRAN" in self.analyses:
            print("\nStarting Transient Analysis...")
            t_stop = self.analyses[".TRAN"]["stop"]
            dt = self.analyses[".TRAN"]["step"]
            
            # Get t=0 starting bias
            _, v_initial = dc_engine.compute_dc_bias(Y_base, sources_base)
            
            # Run Forward Engine
            tran_engine = TransientEngine(self.circuit, self.is_nonlinear, self.ramp)
            time, VI, lus = tran_engine.run(
                Y_base, sources_base, v_initial, t_stop, dt, keep_lus=(keep_lus or sensitivity)
            )
            
            # Store in Data Vault
            result = SimulationResult(".TRAN", time, VI, self.circuit.node_map, dt=dt)
            result.list_of_lus = lus
            
            # Run Backward Engine
            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_transient(time, VI, lus, dt)
                
            return result

        # ==========================================
        # AC ROUTE
        # ==========================================
        elif ".AC" in self.analyses:
            print("\nStarting AC Analysis...")
            start = self.analyses[".AC"]["start"]
            stop = self.analyses[".AC"]["stop"]
            pts = self.analyses[".AC"]["num_points"]
            
            # Extension: Check for linear vs decade sweep type if the parser provides it
            sweep_type = self.analyses[".AC"].get("sweep_type", "DEC")
            
            # 1. Get the DC linearization point (Operating Point)
            _, v_dc = dc_engine.compute_dc_bias(Y_base, sources_base)
            
            # 2. Run AC Engine
            ac_engine = ACEngine(self.circuit, self.is_nonlinear)
            freq, VI, lus = ac_engine.run(
                Y_base, v_dc, start, stop, pts, sweep_type=sweep_type, keep_lus=(keep_lus or sensitivity)
            )
            
            # 3. Store in Data Vault
            result = SimulationResult(".AC", freq, VI, self.circuit.node_map)
            result.list_of_lus = lus

            # Run AC Adjoint Engine
            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(VI, lus, freq_array=freq)
            
                
            return result

        # ==========================================
        # DC SWEEP ROUTE
        # ==========================================
        elif ".DC" in self.analyses:
            # Extract sweep parameters from your parsed dictionary
            source_name = self.analyses[".DC"]["source"]
            start = self.analyses[".DC"]["start"]
            stop = self.analyses[".DC"]["stop"]
            step = self.analyses[".DC"]["step"]
            
            # Run DC Sweep Engine
            sweep_engine = DCSweepEngine(self.circuit, dc_engine)
            sweep_axis, VI, lus = sweep_engine.run(
                Y_base, sources_base, source_name, start, stop, step, keep_lus=(keep_lus or sensitivity)
            )
            
            # Store in Data Vault
            result = SimulationResult(".DC", sweep_axis, VI, self.circuit.node_map)
            result.list_of_lus = lus

            # Run DC Adjoint Engine
            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(VI, lus)
                
            return result
            
        # ==========================================
        # OPERATING POINT (.OP) ROUTE
        # ==========================================
        elif ".OP" in self.analyses:
            print("\nStarting DC Operating Point Analysis...")
            lu_dc, v_dc = dc_engine.compute_dc_bias(Y_base, sources_base)
            
            # Store single point in Data Vault
            result = SimulationResult(".OP", np.array([0.0]), v_dc, self.circuit.node_map)
            result.list_of_lus = [lu_dc]

            # Run OP Adjoint Engine
            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(np.array([v_dc]), [lu_dc])
            
            return result

        else:
            raise ValueError("No recognized analysis (.TRAN, .AC, .DC, .OP) found in the parsed netlist commands.")
