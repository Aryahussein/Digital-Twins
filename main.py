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
import numpy as np


def run_simulation_core(
    netlist_path,
    output_nodes=None,
    sensitivity=False,
    sensitivity_post=False,
    keep_lus=False,
):
    # 1) Parse netlist
    parser = NetlistParser()
    components, analyses = parser.parse(netlist_path)

    node_map = build_node_index(components)

    # 2) Setup simulator (stamping + nonlinearity + solver routing are inside Simulator)
    # mh why fixed ramp
    ramp = 10
    sim = Simulator(components, analyses, node_map, output_nodes, ramp=ramp)

    # 3) Execute requested analysis
    # (frequencies_array, solutions_2d_array, list_of_lus, list_of_sensitivities)
    x_axis, VI, list_of_lus, raw_sens = sim.execute_analysis(
        sensitivity=sensitivity, keep_lus=(keep_lus or sensitivity_post)
    )

    sens_post_proc = raw_sens

    # 4) Sensitivity handling (post-processing or in-solve)
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
    }


if __name__ == "__main__":

    # ==========================================
    # TOGGLE THIS TO SWITCH BETWEEN GUI AND CLI
    USE_GUI = False
    # ==========================================

    if USE_GUI:
        import tkinter as tk
        from gui import CircuitSimulatorGUI  # Make sure gui code is saved as gui.py

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
            help="Folder used when netlist is provided as a name (default: ../testfiles)",
        )
        ap.add_argument(
            "--nodes",
            default=None,
            help="Comma-separated output nodes for adjoint (e.g. 2 or 2,3). If omitted: all nodes.",
        )
        ap.add_argument(
            "--plotnode",
            default=None,
            help="Single node number to plot (e.g. 2). If omitted: use tools default behavior.",
        )
        ap.add_argument(
            "--component",
            default=None,
            help='Component key to plot sensitivity for (e.g. "R1" or "D1:RS" or "M1:VTO"). '
            "If omitted, your plotting function may plot all keys (if implemented).",
        )
        ap.add_argument(
            "--sens",
            action="store_true",
            help="Compute sensitivity during solve (lower memory). Recommended for few nodes.",
        )
        ap.add_argument(
            "--senspost",
            action="store_true",
            help="Compute sensitivity in post-processing across sweep (stores LU). Best for full sweeps.",
        )
        ap.add_argument(
            "--keep_lus",
            action="store_true",
            help="Force keeping LU factors (useful for debugging).",
        )

        args = ap.parse_args()

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
            target_nodes = [int(x.strip()) for x in args.nodes.split(",") if x.strip()]

        # Parse plot node
        plot_node = None if args.plotnode is None else int(args.plotnode)

        # Sensitivity mode: avoid conflicting flags
        sensitivity = bool(args.sens)
        sens_post_proc = bool(args.senspost)
        if sensitivity and sens_post_proc:
            # Prefer post-processing if both are requested
            sensitivity = False

        keep_lus = args.keep_lus or sens_post_proc

        results = run_simulation_core(
            file_path,
            output_nodes=target_nodes,
            sensitivity=sensitivity,
            sensitivity_post=sens_post_proc,
            keep_lus=keep_lus,
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
                folder="../figures/ac",
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
                    folder="../figures/ac",
                    name=f"{netlist_name}_ac_sens",
                )

        elif ".TRAN" in analyses:
            plot_transient(
                x_axis,
                VI,
                node_map,
                plot_node,
                folder="../figures/tran",
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
                    folder="../figures/tran",
                    name=f"{netlist_name}_tran_sens",
                )

        else:
            # OP
            if sens_post_proc or sensitivity:
                print(
                    f"Sensitivity for output nodes {target_nodes if target_nodes is not None else 'ALL'}:"
                )
                print(sensitivities_list)
