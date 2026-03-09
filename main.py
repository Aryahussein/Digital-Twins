import logging
import numpy as np

from parser import NetlistParser
from simulations import Simulator
from node_index import build_node_index
from sensitivity import aggregate_sweep_sensitivities, compute_step_sensitivities
from tools import (
    make_bode_plot,
    plot_ac_sensitivity,
    plot_transient,
    plot_transient_sensitivity,
)

logger = logging.getLogger(__name__)


def run_simulation_core(
    netlist_path,
    output_nodes=None,
    sensitivity=False,
    sensitivity_post=False,
    keep_lus=True,
    run_noise=False,
    run_yield=False,
    run_fault=False,
    spec_min=None,
    spec_max=None,
    noise_freq=1000.0,
):
    """
    Core simulation entry point. Parses the netlist, runs the requested
    analysis, and optionally computes sensitivities and adjoint applications.

    Returns a dict with all simulation results.
    """
    # 1) Parse netlist
    parser = NetlistParser()
    components, analyses = parser.parse(netlist_path)

    node_map = build_node_index(components)

    # 2) Setup simulator
    ramp = 10
    sim = Simulator(components, analyses, node_map, output_nodes, ramp=ramp)

    # If any adjoint application is requested, we need sensitivity
    if run_noise or run_yield or run_fault:
        sensitivity = True

    # 3) Execute requested analysis
    x_axis, VI, list_of_lus, raw_sens = sim.execute_analysis(
        sensitivity=sensitivity, keep_lus=(keep_lus or sensitivity_post)
    )

    # 4) Sensitivity handling
    sens_post_proc = raw_sens
    sens_post_proc_alex = None

    if sensitivity_post or sensitivity:
        if output_nodes is None:
            output_nodes = list(node_map.keys())

        if ".TRAN" in analyses or ".AC" in analyses:
            dt = analyses.get(".TRAN", {}).get("step", None)
            sens_post_proc, sens_post_proc_alex = aggregate_sweep_sensitivities(
                components,
                node_map,
                analyses,
                raw_sensitivities=raw_sens,
                output_nodes=output_nodes,
                list_of_lus=list_of_lus,
                VI_list=VI,
                freq_list=x_axis if ".AC" in analyses else None,
                dt=dt,
            )
        else:
            # OP: single-step sensitivity
            lu0 = (
                list_of_lus[0]
                if isinstance(list_of_lus, (list, tuple))
                else list_of_lus
            )
            VI0 = VI[0] if isinstance(VI, (list, tuple)) else VI

            sens_post_proc, sens_post_proc_alex = compute_step_sensitivities(
                lu0,
                VI0,
                components,
                node_map,
                output_nodes,
                w=(analyses.get(".OP", {}).get("freq", 0.0) * 2 * np.pi),
            )

    # 5) Adjoint applications (noise, yield, fault dictionary)
    adjoint_results = None
    if (run_noise or run_yield or run_fault) and ".OP" in analyses:
        from adjoint_applications import run_adjoint_applications

        # Get the LU and VI for OP
        lu0 = (
            list_of_lus[0]
            if isinstance(list_of_lus, (list, tuple))
            else list_of_lus
        )
        VI0 = VI[0] if isinstance(VI, (list, tuple)) else VI

        # Pick the output node for adjoint applications
        # If user specified nodes, use the first integer one.
        # Otherwise default to the highest-numbered voltage node
        # (typically the output in simple circuits).
        int_nodes = sorted([k for k in node_map if isinstance(k, int)])
        if output_nodes:
            adj_candidates = sorted([n for n in output_nodes if isinstance(n, int)])
            adj_node = adj_candidates[-1] if adj_candidates else int_nodes[-1]
        else:
            adj_node = int_nodes[-1] if int_nodes else 1

        # Get sensitivities for this node
        node_sens = sens_post_proc.get(adj_node, {}) if sens_post_proc else None

        adjoint_results = run_adjoint_applications(
            components,
            node_map,
            VI0,
            lu0,
            adj_node,
            sensitivities=node_sens,
            spec_min=spec_min,
            spec_max=spec_max,
            freq=noise_freq,
        )

    return {
        "analyses": analyses,
        "components": components,
        "node_map": node_map,
        "x_axis": x_axis,
        "VI": VI,
        "sens_post_proc": sens_post_proc,
        "sens_post_proc_alex": sens_post_proc_alex,
        "output_nodes": output_nodes,
        "list_of_lus": list_of_lus,
        "adjoint": adjoint_results,
    }


if __name__ == "__main__":

    # ==========================================
    # TOGGLE THIS TO SWITCH BETWEEN GUI AND CLI
    USE_GUI = True
    # ==========================================

    # Configure logging (console output for both modes)
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    if USE_GUI:
        import tkinter as tk
        from gui import CircuitSimulatorGUI

        root = tk.Tk()
        app = CircuitSimulatorGUI(root, run_simulation_core)
        root.mainloop()

    else:
        import argparse

        ap = argparse.ArgumentParser(
            description="Run circuit simulation + optional sensitivity."
        )
        ap.add_argument(
            "netlist",
            help="Path to netlist .txt OR a testfiles name like RS_latch (without .txt)",
        )
        ap.add_argument(
            "--testdir",
            default="testfiles",
            help="Folder used when netlist is provided as a name (default: testfiles)",
        )
        ap.add_argument(
            "--nodes",
            default=None,
            help="Comma-separated output nodes for adjoint (e.g. 2 or 2,3). "
            "If omitted: all nodes.",
        )
        ap.add_argument(
            "--plotnode",
            default=None,
            help="Single node number to plot (e.g. 2). If omitted: use default.",
        )
        ap.add_argument(
            "--component",
            default=None,
            help='Component key to plot sensitivity for (e.g. "R1" or "D1:RS"). '
            "If omitted, plots top components.",
        )
        ap.add_argument(
            "--sens",
            action="store_true",
            help="Compute sensitivity during solve (lower memory).",
        )
        ap.add_argument(
            "--senspost",
            action="store_true",
            help="Compute sensitivity in post-processing (stores LU factors).",
        )
        ap.add_argument(
            "--keep_lus",
            action="store_true",
            help="Force keeping LU factors (useful for debugging).",
        )
        ap.add_argument(
            "--noise",
            action="store_true",
            help="Run noise analysis (OP only).",
        )
        ap.add_argument(
            "--noise_freq",
            type=float,
            default=1000.0,
            help="Frequency for noise analysis (default: 1000 Hz).",
        )
        ap.add_argument(
            "--yield_analysis",
            action="store_true",
            help="Run yield estimation (requires --spec_min and --spec_max).",
        )
        ap.add_argument(
            "--fault",
            action="store_true",
            help="Build fault dictionary (OP only).",
        )
        ap.add_argument(
            "--spec_min",
            type=float,
            default=None,
            help="Lower specification limit for yield/design centering.",
        )
        ap.add_argument(
            "--spec_max",
            type=float,
            default=None,
            help="Upper specification limit for yield/design centering.",
        )
        ap.add_argument(
            "--verbose",
            action="store_true",
            help="Enable debug-level logging.",
        )

        args = ap.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Resolve netlist path
        if args.netlist.endswith(".txt"):
            file_path = args.netlist
            netlist_name = args.netlist.split("/")[-1].replace(".txt", "")
        else:
            netlist_name = args.netlist
            file_path = f"{args.testdir}/{args.netlist}.txt"

        # Parse target nodes
        if args.nodes is None:
            target_nodes = None
        else:
            target_nodes = [
                int(x.strip()) for x in args.nodes.split(",") if x.strip()
            ]

        # Parse plot node
        plot_node = None if args.plotnode is None else int(args.plotnode)

        # Sensitivity mode
        sensitivity = bool(args.sens)
        sens_post_proc = bool(args.senspost)
        if sensitivity and sens_post_proc:
            sensitivity = False  # prefer post-processing if both requested

        keep_lus = args.keep_lus or sens_post_proc

        results = run_simulation_core(
            file_path,
            output_nodes=target_nodes,
            sensitivity=sensitivity,
            sensitivity_post=sens_post_proc,
            keep_lus=keep_lus,
            run_noise=args.noise,
            run_yield=args.yield_analysis,
            run_fault=args.fault,
            spec_min=args.spec_min,
            spec_max=args.spec_max,
            noise_freq=args.noise_freq,
        )

        analyses = results["analyses"]
        x_axis = results["x_axis"]
        VI = results["VI"]
        node_map = results["node_map"]
        sensitivities_list = results["sens_post_proc"]

        # Visualize
        if ".AC" in analyses:
            make_bode_plot(
                x_axis,
                VI,
                node_map,
                plot_node,
                folder="./figures/ac",
                name=f"{netlist_name}_bode",
            )
            if sens_post_proc or sensitivity:
                plot_ac_sensitivity(
                    x_axis,
                    VI,
                    sensitivities_list,
                    node_map,
                    plot_node,
                    target_component=args.component,
                    folder="./figures/ac",
                    name=f"{netlist_name}_ac_sens",
                )

        elif ".TRAN" in analyses:
            plot_transient(
                x_axis,
                VI,
                node_map,
                plot_node,
                folder="./figures/tran",
                name=f"{netlist_name}_tran",
            )
            if sens_post_proc or sensitivity:
                plot_transient_sensitivity(
                    x_axis,
                    VI,
                    sensitivities_list,
                    node_map,
                    plot_node,
                    target_component=args.component,
                    folder="./figures/tran",
                    name=f"{netlist_name}_tran_sens",
                )

        else:
            # OP
            if sens_post_proc or sensitivity:
                logger.info(
                    "Sensitivity for output nodes %s:",
                    target_nodes if target_nodes is not None else "ALL",
                )
                logger.info("%s", sensitivities_list)

        # Print adjoint application results
        adj = results.get("adjoint")
        if adj:
            print("\n" + "=" * 60)
            print("ADJOINT APPLICATIONS")
            print("=" * 60)

            if "noise" in adj:
                n = adj["noise"]
                print(f"\nNoise at {args.noise_freq:.0f} Hz:")
                print(f"  Total: {n['total_Vrms_per_rtHz']:.4e} V/√Hz")
                for comp, val in sorted(
                    n["contributions"].items(), key=lambda x: -x[1]
                ):
                    if val > 0:
                        pct = val / n["total_V2_per_Hz"] * 100
                        print(
                            f"  {comp:>8s}: {np.sqrt(val):.4e} V/√Hz  ({pct:.1f}%)"
                        )

            if "yield" in adj:
                y = adj["yield"]
                print(f"\nYield Estimation:")
                print(f"  σ_output   = {y['sigma_output']:.4e}")
                print(f"  Yield      = {y['yield_estimate']*100:.4f}%")
                print(f"  Cpk        = {y['cpk']:.2f}")
                print(f"  Defects    = {y['yield_ppm']:.1f} ppm")
                if y["top_contributors"]:
                    print(f"  Top contributors:")
                    for name, pct in y["top_contributors"][:5]:
                        print(f"    {name}: {pct:.1f}% of σ")

            if "fault_dictionary" in adj:
                fd = adj["fault_dictionary"]
                n_entries = sum(len(v) for v in fd.values())
                print(f"\nFault Dictionary: {len(fd)} components, {n_entries} entries")
                # Print top 5 most sensitive
                all_faults = []
                for comp, entries in fd.items():
                    for e in entries:
                        all_faults.append((comp, e))
                all_faults.sort(key=lambda x: -abs(x[1]["delta_output"]))
                print(f"  Top faults by output impact:")
                for comp, e in all_faults[:8]:
                    print(
                        f"    {comp:>8s} {e['fault']:>30s} → ΔV = {e['delta_output']:+.4e}"
                    )

            if "design_centering" in adj:
                dc = adj["design_centering"]
                print(f"\nDesign Centering (recommended adjustments):")
                sorted_dc = sorted(dc.items(), key=lambda x: -abs(x[1]))
                for name, delta in sorted_dc[:5]:
                    print(f"    {name}: Δ = {delta:+.4e}")
