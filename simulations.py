import logging
import numpy as np
from sources import evaluate_all_time_sources, build_ac_sources
from solver import solve_nonlinear_circuit, solve_linear_circuit
from assembleYmatrix import (
    initialize_stamps,
    stamp_source_components,
    stamp_mna_connections,
    stamp_dynamic_components,
    stamp_transient_components,
    stamp_nonlinear_components,
    stamp_static_components,
)
from sensitivity import compute_step_sensitivities
from tools import print_solution

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, components, analyses, node_map, output_nodes=None, ramp=1):
        self.components = components
        self.analyses = analyses
        self.node_map = node_map
        self.output_nodes = output_nodes
        self.total_dim = len(node_map)
        self.ramp = ramp

        # Centralized nonlinearity check
        self.is_nonlinear = any(
            comp["type"] in ["D", "M", "Q", "O"] for comp in components.values()
        )

        # Automatic matrix type determination
        self.is_complex = self._check_if_complex_needed()
        logger.info("Matrix type: %s", "Complex" if self.is_complex else "Real")

        # Initialize Base Matrices
        self.Y_base, self.sources_base = initialize_stamps(
            self.total_dim, is_complex=self.is_complex
        )

        # Stamp MNA Connections (Topology only)
        stamp_mna_connections(self.Y_base, self.components, self.node_map)

        # Stamp Static Components (R, G)
        stamp_static_components(
            self.Y_base, self.sources_base, self.components, self.node_map
        )

        # Finalize base matrix
        self.Y_base = self.Y_base.tocsc()

    def _check_if_complex_needed(self):
        """Internal check to see if we need complex numbers for AC."""
        if any(k in self.analyses for k in [".AC", ".ac"]):
            return True
        if ".OP" in self.analyses and self.analyses[".OP"].get("freq", 0.0) > 0.0:
            return True
        return False

    def _get_dc_bias(self, evaluated_components=None):
        """
        Calculates the steady-state DC operating point.

        Treats all capacitors as open circuits and inductors as short circuits.
        If the circuit is non-linear, it initiates the Newton-Raphson solver.
        """
        comps = (
            evaluated_components
            if evaluated_components is not None
            else self.components
        )

        logger.debug("Computing DC bias point...")

        Y_dc = self.Y_base.copy()
        sources_dc = self.sources_base.copy()

        stamp_source_components(Y_dc, sources_dc, comps, self.node_map)
        stamp_dynamic_components(Y_dc, sources_dc, comps, self.node_map, w=0.0)

        if self.is_nonlinear:
            # Build a better initial guess by solving the linear-only circuit.
            # Nonlinear devices are ignored (open-circuited), giving us the
            # source/resistor voltages as a starting point.
            try:
                from scipy.sparse import diags as sp_diags
                n = len(self.node_map)
                gmin_big = np.zeros(n)
                for key, idx in self.node_map.items():
                    if isinstance(key, int):
                        gmin_big[idx] = 1e-6
                Y_linear = Y_dc + sp_diags(gmin_big, 0, shape=(n, n), format="csc")
                V_ini = solve_linear_circuit(Y_linear, sources_dc.copy())[1]
                V_ini = np.real(V_ini)
                # Clip to prevent absurd initial junction voltages
                V_ini = np.clip(V_ini, -100.0, 100.0)
            except Exception:
                V_ini = np.zeros_like(sources_dc)

            return solve_nonlinear_circuit(
                Y_dc,
                sources_dc,
                comps,
                self.node_map,
                V_ini,
                max_iter=100,
                num_steps=self.ramp,
            )
        return solve_linear_circuit(Y_dc, sources_dc)

    def _solve_single_ac_point(self, w, VI_dc, ac_sources):
        """Solves one AC frequency point."""
        Y_ac = self.Y_base.astype(complex)
        sources_step = ac_sources.copy()

        stamp_dynamic_components(
            Y_ac, sources_step, self.components, self.node_map, w=w
        )

        if self.is_nonlinear:
            dummy_dc = np.zeros_like(sources_step)
            stamp_nonlinear_components(
                Y_ac,
                dummy_dc,
                self.components,
                self.node_map,
                v_prev=VI_dc,
                v_guess=VI_dc,
            )

        return solve_linear_circuit(Y_ac, sources_step)

    def _solve_single_time_step(self, comp_t, dt, v_prev):
        """Solves one transient time step."""
        Y_step = self.Y_base.copy()
        sources_step = self.sources_base.copy()

        stamp_transient_components(
            Y_step, sources_step, comp_t, self.node_map, dt, v_prev
        )

        if self.is_nonlinear:
            return solve_nonlinear_circuit(
                Y_step,
                sources_step,
                comp_t,
                self.node_map,
                v_prev,
                max_iter=100,
                num_steps=self.ramp,
                print_stuff=False,
            )
        return solve_linear_circuit(Y_step, sources_step)

    # =========================================================================
    # PUBLIC ANALYSIS METHODS
    # =========================================================================
    def run_op(self, w=0.0, output_nodes=None, sensitivity=False):
        """
        Executes a DC Operating Point analysis or single-point AC analysis.

        Returns:
            tuple: (solution_vector, lu_factorization, sensitivities_dict)
        """
        lu, VI = self._get_dc_bias()

        if w > 0.0:
            ac_sources = build_ac_sources(self.components, self.node_map)
            lu, VI = self._solve_single_ac_point(w, VI, ac_sources)

        print_solution(VI, self.node_map, w=w)

        step_sens, step_sens_alex = (
            compute_step_sensitivities(
                lu, VI, self.components, self.node_map, output_nodes, w=w
            )
            if sensitivity
            else (None, None)
        )
        return VI, lu, step_sens

    def run_ac_sweep(
        self,
        start_freq=10,
        stop_freq=100000,
        points=100,
        output_nodes=None,
        keep_lus=False,
        sensitivity=False,
    ):
        """
        Executes an AC small-signal frequency sweep.

        Returns:
            tuple: (frequencies_array, solutions_2d_array, list_of_lus, list_of_sensitivities)
        """
        frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)
        VIs = []
        list_of_lus = [] if keep_lus else None
        list_of_sensitivities = [] if sensitivity else None

        ac_sources = build_ac_sources(self.components, self.node_map)
        _, VI_dc = self._get_dc_bias()

        for f in frequencies:
            w = 2 * np.pi * f
            lu_ac, VI_ac = self._solve_single_ac_point(w, VI_dc, ac_sources)

            VIs.append(VI_ac)
            if keep_lus:
                list_of_lus.append(lu_ac)
            if sensitivity:
                # FIX: renamed to avoid shadowing the `sensitivity` boolean
                step_sens, _ = compute_step_sensitivities(
                    lu_ac,
                    VI_ac,
                    self.components,
                    self.node_map,
                    output_nodes,
                    w=w,
                )
                list_of_sensitivities.append(step_sens)

        return frequencies, np.array(VIs), list_of_lus, list_of_sensitivities

    def run_transient(
        self, t_stop, dt, output_nodes=None, keep_lus=False, sensitivity=False
    ):
        """
        Executes a time-domain transient simulation using Backward Euler.

        Returns:
            tuple: (time_array, solutions_2d_array, list_of_lus, list_of_sensitivities)
        """
        time_array = np.arange(0, t_stop, dt)
        results = np.zeros((len(time_array), self.total_dim))
        list_of_lus = [] if keep_lus else None
        list_of_sensitivities = [] if sensitivity else None

        # Calculate initial conditions at t=0
        comp_t0 = evaluate_all_time_sources(self.components, 0.0)
        _, v_prev = self._get_dc_bias(evaluated_components=comp_t0)

        logger.info("Initial Conditions computed. Starting time loop...")

        for step, t in enumerate(time_array):
            logger.debug("Solving at time %.6e", t)
            comp_t = evaluate_all_time_sources(self.components, t)
            lu, VI = self._solve_single_time_step(comp_t, dt, v_prev)

            results[step, :] = VI
            v_prev = VI

            if keep_lus:
                list_of_lus.append(lu)
            if sensitivity:
                # FIX: renamed to avoid shadowing the `sensitivity` boolean
                step_sens, _ = compute_step_sensitivities(
                    lu, VI, self.components, self.node_map, output_nodes, dt=dt
                )
                list_of_sensitivities.append(step_sens)

        return time_array, results, list_of_lus, list_of_sensitivities

    def execute_analysis(self, sensitivity=False, keep_lus=False):
        """
        The Master Router. Decides which simulation loop to run
        based on the parsed netlist commands.
        """
        if ".TRAN" in self.analyses:
            logger.info("Running transient analysis...")
            t_stop = self.analyses[".TRAN"]["stop"]
            dt = self.analyses[".TRAN"]["step"]
            return self.run_transient(
                t_stop, dt, self.output_nodes, keep_lus, sensitivity
            )

        elif ".AC" in self.analyses:
            logger.info("Running AC analysis...")
            a = self.analyses[".AC"]
            return self.run_ac_sweep(
                a["start"],
                a["stop"],
                a["num_points"],
                self.output_nodes,
                keep_lus,
                sensitivity,
            )

        else:  # Default to .OP
            logger.info("Running OP analysis...")
            freq = self.analyses.get(".OP", {}).get("freq", 0.0)
            w = freq * 2 * np.pi
            VI, lu, sens = self.run_op(
                w=w, output_nodes=self.output_nodes, sensitivity=sensitivity
            )
            return None, VI, lu, sens
