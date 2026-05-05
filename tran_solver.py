import numpy as np
import matplotlib.pyplot as plt

from dc_solver import run_dc

_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def run_tran(components, node_index, N, Mv, Mo,
             dt, tstop,
             sens_node=None,
             print_requests=None):

    if dt <= 0:
        raise RuntimeError("Transient timestep must be > 0")

    steps = max(1, int(np.ceil(tstop / dt)))
    size  = N + Mv + Mo

    print("Computing DC operating point...")
    x_prev = run_dc(components, node_index, N, Mv, Mo)

    history  = []
    time_vec = []

    if not print_requests:
        print("Warning: No .print specified — nothing will be plotted")
        print_requests = []

    outputs = [[[] for _ in node_list] for _, node_list in print_requests]

    for step in range(steps + 1):

        t = step * dt
        x = x_prev.copy()

        for _ in range(100):

            G = np.zeros((size, size))
            b = np.zeros(size)

            for i in range(size):
                G[i, i] += 1e-12

            ctx = {
                "node_index": node_index,
                "analysis":   "tran",
                "dt":         dt,
                "x_prev":     x_prev,
                "x":          x,
                "N":          N,
                "Mv":         Mv,
                "t":          t
            }

            for comp in components:
                comp.stamp(G, b, ctx)

            x_new = np.linalg.solve(G, b)

            if np.max(np.abs(x_new - x)) < 1e-6:
                break

            x = x_new

        x = x_new
        x_prev = x.copy()

        history.append(x.copy())
        time_vec.append(t)

        for i, (req_type, node_list) in enumerate(print_requests):
            for j, node in enumerate(node_list):

                if req_type.lower() == 'v':
                    outputs[i][j].append(
                        x[node_index[node]] if node in node_index else np.nan
                    )

                elif req_type.lower() == 'i':
                    from MODELS.voltage_source import VoltageSource
                    val = np.nan
                    for comp in components:
                        if isinstance(comp, VoltageSource) and comp.name.lower() == node.lower():
                            val = abs(x[N + comp.index])
                            break
                    outputs[i][j].append(val)

                else:
                    outputs[i][j].append(np.nan)

    print("\n===== TRANSIENT DONE =====")

    if print_requests:
        fig, axes = plt.subplots(len(print_requests), 1, sharex=True,
                                 figsize=(10, 3 * len(print_requests)))

        if len(print_requests) == 1:
            axes = [axes]

        for i, (req_type, node_list) in enumerate(print_requests):
            ax = axes[i]

            for j, node in enumerate(node_list):
                ax.plot(time_vec, outputs[i][j],
                        color=_COLORS[j % len(_COLORS)],
                        label=f"{req_type.upper()}({node})")

            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    return history