import numpy as np
import matplotlib.pyplot as plt


def run_ac(components, node_index, N, Mv, Mo, frequencies,
           sens_node=None, print_requests=None, x_op=None):

    size = N + Mv + Mo

    if x_op is None:
        raise RuntimeError("AC analysis requires DC operating point (x_op)")

    results = []
    outputs = []

    # ================================
    # Frequency Sweep
    # ================================
    for f in frequencies:

        w = 2 * np.pi * f
        jw = 1j * w

        G = np.zeros((size, size), dtype=complex)
        b = np.zeros(size, dtype=complex)

        ctx = {
            "node_index": node_index,
            "analysis": "ac",
            "jw": jw,
            "N": N,
            "Mv": Mv,
            "x": x_op   # 🔥 key change (linearization point)
        }

        for comp in components:
            comp.stamp(G, b, ctx)

        try:
            X = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            raise RuntimeError("AC matrix singular — check circuit")

        results.append(X)

        # -------- PRINT COLLECTION --------
        if print_requests:
            row_vals = []
            for req_type, node in print_requests:
                if req_type.lower() == 'v':
                    row_vals.append(X[node_index[node]])
            outputs.append(row_vals)

    print("\n===== AC DONE =====")
    print("Frequency points:", len(frequencies))

    # ================================
    # Plotting
    # ================================
    if print_requests:

        outputs = np.array(outputs)

        for i, (req_type, node) in enumerate(print_requests):

            vals = outputs[:, i]

            mag = 20 * np.log10(np.maximum(np.abs(vals), 1e-30))
            phase = np.angle(vals, deg=True)

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            ax1.semilogx(frequencies, mag)
            ax1.set_ylabel("Magnitude (dB)")
            ax1.set_title(f"AC Response V({node})")
            ax1.grid(True, which="both")

            ax2.semilogx(frequencies, phase)
            ax2.set_ylabel("Phase (deg)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.grid(True, which="both")

            plt.tight_layout()
            plt.show()

    return results