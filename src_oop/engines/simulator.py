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
        analyses (dict[str, dict]): A dictionary of requested analyses parsed
            from the netlist (e.g., {'.TRAN': {'stop': 1e-3, 'step': 1e-6}}).
        output_nodes (list[str or int]): Target nodes for Adjoint sensitivity
            calculations. Defaults to all nodes in the circuit.
        is_nonlinear (bool): Automatically detected flag indicating if
            Newton-Raphson solvers are required by any component.
        is_complex (bool): Automatically detected flag indicating if complex
            matrix dtypes are required for AC analysis.
    """

    def __init__(self, circuit, analyses, output_nodes=None):
        """Initializes the Simulator environment.

        Args:
            circuit (Circuit): The constructed circuit object.
            analyses (dict[str, dict]): The dictionary of simulation commands.
            output_nodes (list[str or int], optional): Specific nodes to track
                for sensitivity. Defaults to None, which automatically tracks
                all valid voltage nodes.
        """
        self.circuit = circuit
        self.analyses = analyses
        self.output_nodes = (
            output_nodes if output_nodes else list(circuit.node_map.keys())
        )

        # Check flags by peeking into the object types
        self.is_nonlinear = any(comp.IS_NONLINEAR for comp in circuit.components)
        self.is_complex = ".AC" in analyses

    def execute_analysis(self, sensitivity=False, global_adjoint=False, keep_lus=False):
        """Routes and executes the requested simulation.

        This method builds the static matrix topology once, delegates mathematical
        solving to the specific forward engine (.TRAN, .AC, .DC, .OP), unifies
        the output formatting, and optionally triggers the backward Adjoint passes.

        Args:
            sensitivity (bool, optional): If True, triggers the calculation of the
                unified continuous sensitivity tensor. Defaults to False.
            global_adjoint (bool, optional): If True, triggers the backward-propagating
                global sensitivity integral. This flag only applies to .TRAN analyses.
                Defaults to False.
            keep_lus (bool, optional): If True, caches the scipy LU factorizations
                in memory. This is automatically enabled if either sensitivity flag
                is True. Defaults to False.

        Returns:
            SimulationResult: A secure data vault containing all forward state
            variables and any requested sensitivity gradients.

        Raises:
            ValueError: If no recognized analysis command (.TRAN, .AC, .DC, .OP)
            is found in the analyses dictionary.
        """
        # 1. Base Topology Setup
        dc_engine = DCEngine(self.circuit, self.is_complex, self.is_nonlinear)
        Y_base = dc_engine.build_base_matrices()

        # Define universal payload variables for clean packaging
        analysis_type = ""
        domain = ""
        sweep_axis = None
        VI = None
        lus = None
        dt = 0.0
        method = "BE"  # Default integration method

        # ==========================================
        # 2. RUN THE FORWARD MATH ENGINES
        # ==========================================
        if ".TRAN" in self.analyses:
            print("\nStarting Transient Analysis...")
            t_stop = self.analyses[".TRAN"]["stop"]
            dt = self.analyses[".TRAN"]["step"]
            method = self.analyses.get("OPTIONS", {}).get("method", method)

            _, v_initial = dc_engine.compute_dc_bias(Y_base)
            tran_engine = TransientEngine(self.circuit, self.is_nonlinear)

            sweep_axis, VI, lus = tran_engine.run(
                Y_base,
                v_initial,
                t_stop,
                dt,
                keep_lus=(keep_lus or sensitivity or global_adjoint),
                method=method,
            )
            analysis_type, domain = ".TRAN", "time"

        elif ".AC" in self.analyses:
            print("\nStarting AC Analysis...")
            start = self.analyses[".AC"]["start"]
            stop = self.analyses[".AC"]["stop"]
            pts = self.analyses[".AC"]["num_points"]
            sweep_type = self.analyses[".AC"].get("sweep_type", "DEC")

            _, v_dc = dc_engine.compute_dc_bias(Y_base)
            ac_engine = ACEngine(self.circuit, self.is_nonlinear)

            sweep_axis, VI, lus = ac_engine.run(
                Y_base,
                v_dc,
                start,
                stop,
                pts,
                sweep_type=sweep_type,
                keep_lus=(keep_lus or sensitivity),
            )
            analysis_type, domain = ".AC", "frequency"

        elif ".DC" in self.analyses:
            print("\nStarting DC Sweep Analysis...")
            source = self.analyses[".DC"]["source"]
            start = self.analyses[".DC"]["start"]
            stop = self.analyses[".DC"]["stop"]
            step = self.analyses[".DC"]["step"]

            sweep_axis, VI, lus = dc_engine.compute_dc_sweep(
                Y_base, source, start, stop, step, keep_lus=(keep_lus or sensitivity)
            )
            analysis_type, domain = ".DC", "voltage"

        elif ".OP" in self.analyses:
            print("\nStarting DC Operating Point Analysis...")
            lu_dc, v_dc = dc_engine.compute_dc_bias(Y_base)

            # Wrap standard 1D .OP output into sweep-compatible formats
            sweep_axis = np.array([0.0])
            VI = np.array([v_dc])
            lus = [lu_dc]
            analysis_type, domain = ".OP", "static"

        else:
            raise ValueError(
                "No recognized analysis (.TRAN, .AC, .DC, .OP) found in the parsed netlist commands."
            )

        # ==========================================
        # 3. PACKAGE FORWARD RESULTS
        # ==========================================
        result = SimulationResult(analysis_type, sweep_axis, VI, self.circuit.node_map)
        result.list_of_lus = lus

        # ==========================================
        # 4. RUN THE ADJOINT ENGINES
        # ==========================================
        if sensitivity or global_adjoint:
            adj_engine = AdjointEngine(self.circuit, self.output_nodes)

            # 4A. Run the Unified Continuous Tensor (For all domains)
            if sensitivity:
                result.sensitivities = adj_engine.compute_sensitivities(
                    sweep_axis=sweep_axis,
                    V_forward=VI,
                    list_of_lus=lus,
                    domain=domain,
                    dt=dt,
                    method=method,
                )

            # 4B. Run the Backward Global Integral (ONLY for TRAN)
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
                    "\nWarning: 'global_adjoint' was requested, but it only applies to .TRAN analyses. Skipping."
                )

        return result
