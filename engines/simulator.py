"""
Master Simulator Module.

Routes analyses to the correct engine: DC, AC, Transient, DC Sweep, OP.
Includes TR + nonlinear adjoint fix: runs a parallel BE pass for sensitivity
when TR is selected on a circuit containing MOSFETs or diodes.
"""

from engines.dc_engine import DCEngine
from engines.transient_engine import TransientEngine
from engines.ac_engine import ACEngine
from engines.adjoint_engine import AdjointEngine
from core.results import SimulationResult
import numpy as np


class Simulator:
    def __init__(self, circuit, analyses, output_nodes=None):
        self.circuit = circuit
        self.analyses = analyses
        self.output_nodes = output_nodes if output_nodes else list(circuit.node_map.keys())
        self.is_nonlinear = any(comp.IS_NONLINEAR for comp in circuit.components)
        self.is_complex = ".AC" in analyses

    def execute_analysis(self, sensitivity=False, keep_lus=False, method='BE'):
        """Routes and executes the requested simulation.

        Args:
            sensitivity (bool): Run adjoint sensitivity pass.
            keep_lus   (bool): Cache LU factorizations.
            method     (str) : 'BE' or 'TR'.

        Returns:
            SimulationResult
        """
        dc_engine = DCEngine(self.circuit, self.is_complex, self.is_nonlinear)
        Y_base    = dc_engine.build_base_matrices()

        # ── TRANSIENT ────────────────────────────────────────────────────
        if ".TRAN" in self.analyses:
            print("\nStarting Transient Analysis...")
            t_stop = self.analyses[".TRAN"]["stop"]
            dt     = self.analyses[".TRAN"]["step"]

            _, v_initial = dc_engine.compute_dc_bias(Y_base)

            tran_engine = TransientEngine(self.circuit, self.is_nonlinear, method=method)
            time, VI, lus = tran_engine.run(
                Y_base, v_initial, t_stop, dt, keep_lus=(keep_lus or sensitivity)
            )

            result = SimulationResult(".TRAN", time, VI, self.circuit.node_map, dt=dt)
            result.list_of_lus = lus

            if sensitivity:
                # TR + nonlinear: run a parallel BE pass for the adjoint.
                # TR waveform is kept for display; sensitivity uses BE LUs.
                sens_method = method
                sens_VI     = VI
                sens_lus    = lus

                if method == 'TR' and self.is_nonlinear:
                    print("\n[Sensitivity] TR + nonlinear: running parallel BE pass...")
                    for comp in self.circuit.components:
                        comp.reset_transient_state()
                    be_engine = TransientEngine(self.circuit, self.is_nonlinear, method='BE')
                    _, VI_be, lus_be = be_engine.run(
                        Y_base, v_initial, t_stop, dt, keep_lus=True
                    )
                    sens_VI     = VI_be
                    sens_lus    = lus_be
                    sens_method = 'BE'
                    print("[Sensitivity] BE pass complete.")

                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_transient(
                    time, sens_VI, sens_lus, dt, method=sens_method
                )

            return result

        # ── AC ───────────────────────────────────────────────────────────
        elif ".AC" in self.analyses:
            print("\nStarting AC Analysis...")
            start      = self.analyses[".AC"]["start"]
            stop       = self.analyses[".AC"]["stop"]
            pts        = self.analyses[".AC"]["num_points"]
            sweep_type = self.analyses[".AC"].get("sweep_type", "DEC")

            _, v_dc = dc_engine.compute_dc_bias(Y_base)

            ac_engine = ACEngine(self.circuit, self.is_nonlinear)
            freq, VI, lus = ac_engine.run(
                Y_base, v_dc, start, stop, pts,
                sweep_type=sweep_type, keep_lus=(keep_lus or sensitivity)
            )

            result = SimulationResult(".AC", freq, VI, self.circuit.node_map)
            result.list_of_lus = lus

            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(VI, lus, freq_array=freq)

            return result

        # ── DC SWEEP ─────────────────────────────────────────────────────
        elif ".DC" in self.analyses:
            print("\nStarting DC Sweep Analysis...")
            source_name = self.analyses[".DC"]["source"]
            start       = self.analyses[".DC"]["start"]
            stop        = self.analyses[".DC"]["stop"]
            step        = self.analyses[".DC"]["step"]

            sweep_axis, VI, lus = dc_engine.compute_dc_sweep(
                Y_base, source_name, start, stop, step,
                keep_lus=(keep_lus or sensitivity)
            )

            result = SimulationResult(".DC", sweep_axis, VI, self.circuit.node_map)
            result.list_of_lus = lus

            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(VI, lus)

            return result

        # ── OPERATING POINT ──────────────────────────────────────────────
        elif ".OP" in self.analyses:
            print("\nStarting DC Operating Point Analysis...")

            lu_dc, v_dc = dc_engine.compute_dc_bias(Y_base)

            result = SimulationResult(".OP", np.array([0.0]), v_dc, self.circuit.node_map)
            result.list_of_lus = [lu_dc]

            if sensitivity:
                adj_engine = AdjointEngine(self.circuit, self.output_nodes)
                result.sensitivities = adj_engine.compute_sweep(
                    np.array([v_dc]), [lu_dc]
                )

            return result

        else:
            raise ValueError(
                "No recognized analysis (.TRAN, .AC, .DC, .OP) found."
            )
