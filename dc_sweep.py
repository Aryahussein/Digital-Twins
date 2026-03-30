import numpy as np
import matplotlib.pyplot as plt

# Colours for family-of-curves
_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
           '#17becf', '#bcbd22']

GMIN     = 1e-12
MAX_ITER = 150
TOL      = 1e-6


def _newton_dc(components, node_index, N, Mv, Mo,
               x_init, sweep_overrides):
    """
    Self-contained Newton solve for one DC sweep point.
    Does NOT do source stepping — the warm-start x_init should already
    be close enough that stepping isn't needed between adjacent sweep points.

    sweep_overrides: dict {src_name_lower: float}
    """
    size = N + Mv + Mo
    x    = x_init.copy()

    for _ in range(MAX_ITER):

        G = np.zeros((size, size))
        b = np.zeros(size)

        for i in range(size):
            G[i, i] += GMIN

        ctx = {
            "node_index":      node_index,
            "analysis":        "dc",
            "N":               N,
            "Mv":              Mv,
            "x":               x,
            "source_scale":    1.0,
            "sweep_overrides": sweep_overrides,
        }

        for comp in components:
            comp.stamp(G, b, ctx)

        try:
            x_new = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            raise RuntimeError("Singular matrix during DC sweep step")

        err = np.max(np.abs(x_new - x))
        res = np.max(np.abs(G @ x_new - b))

        if err < TOL and res < TOL:
            return x_new

        alpha = 1.0
        if np.max(np.abs((x + (x_new - x))[:N])) > 50.0:
            alpha = 0.5
        x = x + alpha * (x_new - x)

    raise RuntimeError(
        f"DC sweep Newton failed at {sweep_overrides}  (last err={err:.2e})"
    )


def _branch_current(components, node_index, N, x, source_name):
    """Return MNA branch current variable for a named voltage source."""
    from MODELS.voltage_source import VoltageSource
    for comp in components:
        if isinstance(comp, VoltageSource) and comp.name.lower() == source_name.lower():
            return x[N + comp.index]
    raise KeyError(f"Voltage source '{source_name}' not found in netlist")


def run_dc_sweep(components, node_index, N, Mv, Mo,
                 sweep_params, print_requests):
    """
    DC sweep engine.

    sweep_params  — list of dicts parsed from .dc lines:
      {
        'src'  : 'Vgs',       # source to sweep
        'start': 0.0,
        'stop' : 1.2,
        'step' : 0.01,
        'inner': {            # optional family-of-curves
            'src'   : 'Vds',
            'values': [0.2, 0.6, 1.0, 1.2]
        }
      }

    print_requests — list of (req_type, node_list):
      'v' + [node_name]   → plot node voltage vs sweep variable
      'i' + [src_name]    → plot |branch current| through that voltage source (µA)
    """
    from dc_solver import run_dc

    # Warm-start: solve DC at the netlist's nominal source values
    print("Computing warm-start DC operating point...")
    x_op = run_dc(components, node_index, N, Mv, Mo,
                  sens_node=None, print_requests=None)

    for sp in sweep_params:

        src_outer  = sp['src']
        v_start    = sp['start']
        v_stop     = sp['stop']
        v_step     = sp['step']
        inner      = sp.get('inner', None)

        # Build outer sweep vector (handle negative step)
        if v_step == 0:
            raise RuntimeError(f".dc step cannot be zero for source {src_outer}")
        if v_step < 0:
            sweep_vals = np.arange(v_start, v_stop + v_step * 0.5, v_step)
        else:
            sweep_vals = np.arange(v_start, v_stop + v_step * 0.5, v_step)

        # Inner (family of curves) or single curve
        if inner:
            inner_values = inner['values']
            inner_src    = inner['src']
        else:
            inner_values = [None]
            inner_src    = None

        # One figure per .dc block
        n_plots = max(len(print_requests), 1)
        fig, axes = plt.subplots(n_plots, 1,
                                 figsize=(9, 3.5 * n_plots),
                                 squeeze=False)
        axes = axes.flatten()

        for curve_idx, inner_val in enumerate(inner_values):

            color = _COLORS[curve_idx % len(_COLORS)]
            label = (f"{inner_src}={inner_val:.3g}V"
                     if inner_val is not None else src_outer)

            # Per-print_request storage
            curve_data = [[] for _ in print_requests] if print_requests else []

            x = x_op.copy()   # warm-start; tracks across the sweep

            for v_outer in sweep_vals:

                overrides = {src_outer.lower(): v_outer}
                if inner_val is not None:
                    overrides[inner_src.lower()] = inner_val

                try:
                    x = _newton_dc(components, node_index, N, Mv, Mo,
                                   x, overrides)
                except RuntimeError as e:
                    print(f"  Warning: {e} — NaN inserted at {src_outer}={v_outer:.4f}")
                    for lst in curve_data:
                        lst.append(np.nan)
                    continue

                if print_requests:
                    for pi, (req_type, node_list) in enumerate(print_requests):
                        if req_type.lower() == 'v':
                            vals = [x[node_index[nd]]
                                    for nd in node_list if nd in node_index]
                            curve_data[pi].append(np.mean(vals) if vals else np.nan)
                        elif req_type.lower() == 'i':
                            try:
                                I = _branch_current(components, node_index,
                                                    N, x, node_list[0])
                                curve_data[pi].append(abs(I))
                            except KeyError:
                                curve_data[pi].append(np.nan)
                        else:
                            curve_data[pi].append(np.nan)

            # Plot
            if print_requests:
                for pi, (req_type, node_list) in enumerate(print_requests):
                    ax = axes[pi]
                    y  = np.array(curve_data[pi])

                    if req_type.lower() == 'i':
                        y_plot = y * 1e6
                        ylabel = "Id (µA)"
                    else:
                        y_plot = y
                        ylabel = f"V({', '.join(node_list)}) [V]"

                    ax.plot(sweep_vals[:len(y_plot)], y_plot,
                            color=color, label=label, linewidth=1.5)
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.35)

        # Finalise axes
        for ax in axes:
            ax.set_xlabel(f"{src_outer} (V)")
            if len(inner_values) > 1:
                ax.legend(fontsize=8, framealpha=0.7,
                          title=inner_src, title_fontsize=8)

        title = f"DC Sweep — {src_outer}"
        if inner_src:
            title += f"  (family: {inner_src})"
        fig.suptitle(title, fontsize=11)
        plt.tight_layout()

    plt.show()