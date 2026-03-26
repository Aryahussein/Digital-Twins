import numpy as np
import matplotlib.pyplot as plt

from dc_solver import run_dc


# Colours cycled when multiple nodes share one subplot
_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def run_tran(components, node_index, N, Mv, Mo,
             dt, tstop,
             sens_node=None,
             print_requests=None):

    size  = N + Mv + Mo
    steps = int(tstop / dt)

    print("Computing DC operating point...")
    x_prev = run_dc(
        components, node_index, N, Mv, Mo,
        sens_node=None, print_requests=None
    )

    history  = []
    time_vec = []

    # outputs[i][j] = time series for the j-th node in print_requests[i]
    # Each entry in print_requests is (req_type, [node1, node2, ...])
    if print_requests:
        outputs = [[[] for _ in node_list] for _, node_list in print_requests]
    else:
        outputs = None

    max_iters = 100
    tol       = 1e-6

    for step in range(steps + 1):

        t = step * dt
        x = x_prev.copy()

        for iteration in range(max_iters):

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

            try:
                x_new = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                raise RuntimeError(f"Transient matrix singular at t={t:.3e}")

            err = np.max(np.abs(x_new - x))

            if iteration == 0 and step % 100 == 0:
                print(f"[t={t:.3e}] Newton starting, x_max={np.max(np.abs(x)):.4f}")

            if err < tol:
                break

            alpha = 1.0
            if np.max(np.abs((x + (x_new - x))[:N])) > 20.0:
                alpha = 0.5

            x = x + alpha * (x_new - x)

        else:
            raise RuntimeError(
                f"Newton did not converge at t={t:.3e} (last err={err:.3e})"
            )

        x = x_new
        history.append(x.copy())
        x_prev = x.copy()
        time_vec.append(t)

        if print_requests:
            for i, (_, node_list) in enumerate(print_requests):
                for j, node in enumerate(node_list):
                    outputs[i][j].append(x[node_index[node]])

    print("\n===== TRANSIENT DONE =====")
    print("Steps:", len(history))

    # ================================
    # Plot
    # ================================
    if print_requests:

        n_subplots = len(print_requests)
        fig, axes = plt.subplots(n_subplots, 1, sharex=True,
                                 figsize=(10, 3 * n_subplots))

        if n_subplots == 1:
            axes = [axes]

        for i, (req_type, node_list) in enumerate(print_requests):
            ax = axes[i]

            for j, node in enumerate(node_list):
                color = _COLORS[j % len(_COLORS)]
                ax.plot(time_vec, outputs[i][j],
                        color=color,
                        label=f"{req_type.upper()}({node})",
                        linewidth=1.2)

            if len(node_list) > 1:
                # Multiple traces on this subplot — show legend, no single ylabel
                ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
                ax.set_ylabel("Voltage (V)")
            else:
                ax.set_ylabel(f"{req_type.upper()}({node_list[0]})")

            ax.grid(True, alpha=0.4)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    return history