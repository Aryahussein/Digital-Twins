import numpy as np
import matplotlib.pyplot as plt

_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def run_ac(components, node_index, N, Mv, Mo, frequencies,
           sens_node=None, print_requests=None, x_op=None):

    size = N + Mv + Mo

    if x_op is None:
        raise RuntimeError("AC analysis requires DC operating point (x_op)")

    # outputs[i][j] = list of complex values at each frequency
    # for the j-th node of the i-th print request
    if print_requests:
        outputs = [[[] for _ in node_list] for _, node_list in print_requests]
    else:
        outputs = None

    results = []

    for f in frequencies:

        w  = 2 * np.pi * f
        jw = 1j * w

        G = np.zeros((size, size), dtype=complex)
        b = np.zeros(size, dtype=complex)

        ctx = {
            "node_index": node_index,
            "analysis":   "ac",
            "jw":         jw,
            "N":          N,
            "Mv":         Mv,
            "x":          x_op
        }

        for comp in components:
            comp.stamp(G, b, ctx)

        try:
            X = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            raise RuntimeError("AC matrix singular — check circuit")

        results.append(X)

        if print_requests:
            for i, (req_type, node_list) in enumerate(print_requests):
                for j, node in enumerate(node_list):
                    if req_type.lower() == 'v':
                        outputs[i][j].append(X[node_index[node]])

    print("\n===== AC DONE =====")
    print("Frequency points:", len(frequencies))

    if print_requests:

        n_subplots = len(print_requests)

        for i, (req_type, node_list) in enumerate(print_requests):

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                           figsize=(9, 5))

            for j, node in enumerate(node_list):
                vals  = np.array(outputs[i][j])
                mag   = 20 * np.log10(np.maximum(np.abs(vals), 1e-30))
                phase = np.angle(vals, deg=True)
                color = _COLORS[j % len(_COLORS)]
                label = f"{req_type.upper()}({node})"

                ax1.semilogx(frequencies, mag,   color=color, label=label, linewidth=1.2)
                ax2.semilogx(frequencies, phase, color=color, label=label, linewidth=1.2)

            ax1.set_ylabel("Magnitude (dB)")
            ax1.grid(True, which="both", alpha=0.4)

            ax2.set_ylabel("Phase (deg)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.grid(True, which="both", alpha=0.4)

            if len(node_list) > 1:
                ax1.legend(fontsize=8, framealpha=0.7)
                ax2.legend(fontsize=8, framealpha=0.7)
            else:
                ax1.set_title(f"AC Response {req_type.upper()}({node_list[0]})")

            plt.tight_layout()
            plt.show()

    return results