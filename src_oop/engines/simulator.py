"""Simulator Orchestration Module.

This module provides the top-level Simulator environment. It reads the requested
analysis commands (e.g., .TRAN, .AC, .DC, .OP) parsed from the netlist, delegates 
execution to the appropriate mathematical engines, and seamlessly pipes the 
forward simulation data into the Adjoint engines for sensitivity analysis.
"""

import numpy as np
from engines.dc_engine import DCEngine
from engines.transient_engine import TransientEngine
from engines.ac_engine import ACEngine
from engines.adjoint_engine import AdjointEngine
from core.results import SimulationResult

class Simulator:
    """The master orchestrator for circuit simulation.

    Reads the requested analysis types from the parsed netlist and triggers
    the appropriate simulation engines, ultimately packaging the results into
    a secure `SimulationResult` data vault.

    Attributes:
        circuit (Circuit): The fully initialized, polymorphic circuit orchestrator.
        analyses (dict): A dictionary of requested analyses parsed from the 
            netlist (e.g., {'.TRAN': {'stop': 1e-3, 'step': 1e-6}}).
        output_nodes (list[str | int]): Target nodes for Adjoint sensitivity
            calculations. Defaults to tracking all non-ground nodes in the circuit.
    """

    def __init__(self, circuit, analyses, output_nodes=None):
        """Initializes the Simulator environment.

        Args:
            circuit (Circuit): The constructed circuit object.
            analyses (dict): The dictionary of simulation commands.
            output_nodes (list, optional): Specific nodes to track for sensitivity. 
                Defaults to None, which automatically tracks all valid voltage nodes.
        """
        self.circuit = circuit
        self.analyses = analyses
        self.output_nodes = (
            output_nodes if output_nodes else list(circuit.node_map.keys())
        )

    def execute_analysis(self, sensitivity=False, global_adjoint=False, keep_lus=False):
        """Routes and executes the requested simulation.

        Delegates mathematical solving to the specific forward engine (.TRAN, 
        .AC, .DC, .OP), unifies the output formatting, and optionally triggers 
        the backward Adjoint network passes.

        Args:
            sensitivity (bool, optional): If True, triggers the calculation of the
                unified continuous sensitivity tensor. Defaults to False.
            global_adjoint (bool, optional): If True, triggers the backward-propagating
                global sensitivity integral. This flag only applies to .TRAN analyses.
                Defaults to False.
            keep_lus (bool, optional): If True, caches the scipy LU factorizations
                in memory. Automatically enabled if either sensitivity flag is True. 
                Defaults to False.

        Returns:
            SimulationResult: A secure data vault containing all forward state
            variables and any requested sensitivity gradients.

        Raises:
            ValueError: If no recognized analysis command is found in the 
                analyses dictionary.
        """
        # Define universal payload variables for clean packaging
        analysis_type = ""
        domain = ""
        sweep_axis = None
        VI = None
        lus = None
        dt = 0.0
        method = "TR"  # Default numerical integration method

        # Force LU caching if any Adjoint analysis is requested
        cache_matrices = keep_lus or sensitivity or global_adjoint

        # =====================================================================
        # 1. RUN THE FORWARD MATH ENGINES
        # =====================================================================
        
        step = 0.0
        if ".TRAN" in self.analyses:
            analysis_type, domain = ".TRAN", "time"
            t_stop = self.analyses[".TRAN"]["stop"]
            dt = self.analyses[".TRAN"]["step"]
            step = dt
            method = self.analyses.get("OPTIONS", {}).get("method", method)
            
            initial_conditions = None
            if ".IC" in self.analyses:
                initial_conditions = self.analyses[".IC"]

            tran_engine = TransientEngine(self.circuit)
            sweep_axis, VI, lus = tran_engine.compute(
                t_stop=t_stop,
                dt=dt,
                method=method,
                keep_lus=cache_matrices,
                initial_conditions=initial_conditions,
            )

        elif ".AC" in self.analyses:
            analysis_type, domain = ".AC", "frequency"
            start = self.analyses[".AC"]["start"]
            stop = self.analyses[".AC"]["stop"]
            pts = self.analyses[".AC"]["num_points"]
            sweep_type = self.analyses[".AC"].get("sweep_type", "DEC")
            step = 1.0 / pts

            ac_engine = ACEngine(self.circuit)
            sweep_axis, VI, lus = ac_engine.compute(
                start_freq=start,
                stop_freq=stop,
                points=pts,
                sweep_type=sweep_type,
                keep_lus=cache_matrices,
            )

        elif ".DC" in self.analyses:
            analysis_type, domain = ".DC", "voltage"
            source = self.analyses[".DC"]["source"]
            start = self.analyses[".DC"]["start"]
            stop = self.analyses[".DC"]["stop"]
            step = self.analyses[".DC"]["step"]

            dc_engine = DCEngine(self.circuit)
            sweep_axis, VI, lus = dc_engine.compute_dc_sweep(
                source_name=source, 
                start=start, 
                stop=stop, 
                step=step, 
                keep_lus=cache_matrices
            )

        elif ".OP" in self.analyses:
            print("\nStarting DC Operating Point Analysis...")
            analysis_type, domain = ".OP", "static"
            
            dc_engine = DCEngine(self.circuit)
            lu_dc, v_dc = dc_engine.compute_dc_bias()

            # Wrap standard 1D .OP output into sweep-compatible formats
            sweep_axis = np.array([0.0])
            VI = np.array([v_dc])
            lus = [lu_dc]

        else:
            raise ValueError(
                "No recognized analysis (.TRAN, .AC, .DC, .OP) found in the "
                "parsed netlist commands."
            )

        # =====================================================================
        # 2. PACKAGE FORWARD RESULTS
        # =====================================================================
        result = SimulationResult(
            analysis_type=analysis_type, 
            sweep_axis=sweep_axis, 
            VI_matrix=VI, 
            node_map=self.circuit.node_map, 
            domain=domain, 
            step=step, 
            list_of_lus=lus
        )

        # print(lus)
        # print(result.list_of_lus)

        # =====================================================================
        # 3. RUN THE ADJOINT ENGINES (Backward Passes)
        # =====================================================================
        if sensitivity or global_adjoint:
            adj_engine = AdjointEngine(self.circuit, self.output_nodes)

            # 3A. Run the Unified Continuous Tensor (For all domains)
            if sensitivity:
                result.sensitivities = adj_engine.compute_sensitivities(
                    sweep_axis=sweep_axis,
                    V_forward=VI,
                    list_of_lus=lus,
                    domain=domain,
                    dt=dt,
                    method=method,
                )

            # 3B. Run the Backward Global Integral (ONLY valid for TRAN)
            if global_adjoint and domain == "time":
                result.global_sensitivities = adj_engine.compute_transient(
                    time_array=sweep_axis,
                    V_forward=VI,
                    list_of_lus=lus,
                    dt=dt,
                    method=method,
                )
            elif global_adjoint and domain != "time":
                print(
                    "\nWarning: 'global_adjoint' was requested, but it only "
                    "applies to .TRAN time-domain analyses. Skipping."
                )

        return result
