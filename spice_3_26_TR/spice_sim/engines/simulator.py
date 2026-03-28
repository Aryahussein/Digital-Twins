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

    Attributes:
        circuit (Circuit): The fully initialized, polymorphic circuit object.
        analyses (dict): A dictionary of requested analyses parsed from the netlist.
        output_nodes (list): Target nodes for Adjoint sensitivity calculations.
        is_nonlinear (bool): Auto-detected flag for Newton-Raphson requirement.
        is_complex (bool): Auto-detected flag for complex matrix dtype (AC).
    """

    def __init__(self, circuit, analyses, output_nodes=None):
        self.circuit = circuit
        self.analyses = analyses
        self.output_nodes = output_nodes if output_nodes else list(circuit.node_map.keys())
        
        self.is_nonlinear = any(comp.IS_NONLINEAR for comp in circuit.components)
        self.is_complex = ".AC" in analyses

    def execute_analysis(self, sensitivity=False, keep_lus=False, method='BE'):
        """Routes and executes the requested simulation.

        Args:
            sensitivity (bool): Trigger backward Adjoint pass. Defaults to False.
            keep_lus (bool): Cache LU factorizations. Defaults to False.
            method (str): Transient integration method — 'BE' or 'TR'. Defaults to 'BE'.

        Returns:
            SimulationResult: A secure data vault containing the simulation outputs.
        """
        dc_engine = DCEngine(self.circuit, self.is_complex, self.is_nonlinear)
        Y_base = dc_engine.build_base_matrices()

        # ==========================================
        # TRANSIENT ROUTE
        # ==========================================
        if ".TRAN" in self.analyses:
            print("\nStarting Transient Analysis...")
            t_stop = self.analyses[".TRAN"]["stop"]
            dt = self.analyses[".TRAN"]["step"]
            
            _, v_initial = dc_engine.compute_dc_bias(Y_base)
            
            tran_engine = TransientEngine(self.circuit, self.is_nonlinear, method=method)
            time, VI, lus = tran_engine.run(
                Y_base, v_initial, t_stop, dt, keep_lus=(keep_lus or sensitivity)
            )
            
            result = SimulationResult(".TRAN", time, VI, self.circuit.node_map, dt=dt)
            result.list_of_lus = lus
            
            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_transient(
                    time, VI, lus, dt, method=method
                )
                
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
            print("\nStarting DC Sweep Analysis...")
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
            
            result = SimulationResult(".OP", np.array([0.0]), v_dc, self.circuit.node_map)
            result.list_of_lus = [lu_dc]

            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(np.array([v_dc]), [lu_dc])
            
            return result

        else:
            raise ValueError(
                "No recognized analysis (.TRAN, .AC, .DC, .OP) found in the "
                "parsed netlist commands."
            )
