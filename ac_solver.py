import numpy as np
import matplotlib.pyplot as plt

_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def run_ac(components, node_index, N, Mv, Mo, frequencies,
           sens_node=None, print_requests=None, x_op=None,
           diff_gain_request=None):

    size = N + Mv + Mo

    if x_op is None:
        raise RuntimeError("AC analysis requires DC operating point (x_op)")

    if print_requests:
        outputs = [[[] for _ in node_list] for _, node_list in print_requests]
    else:
        outputs = None

    if diff_gain_request is not None:
        vout_diff = []
        vin_excitation = diff_gain_request.get("vin_ac", 1.0)

    results = []

    for f in frequencies:

        w  = 2 * np.pi * f
        jw = 1j * w

        G = np.zeros((size, size), dtype=complex)
        b = np.zeros(size,         dtype=complex)

        ctx = {
            "node_index": node_index,
            "analysis":   "ac",
            "jw":         jw,
            "N":          N,
            "Mv":         Mv,
            "x":          x_op,
        }

        for comp in components:
            comp.stamp(G, b, ctx)

        try:
            X = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"AC matrix singular at f={f:.3e} Hz — check for floating nodes"
            )

        results.append(X)

        if print_requests:
            for i, (req_type, node_list) in enumerate(print_requests):
                for j, node in enumerate(node_list):
                    if req_type.lower() == 'v':
                        outputs[i][j].append(X[node_index[node]])

        if diff_gain_request is not None:
            out_pos = diff_gain_request["out_pos"]
            out_neg = diff_gain_request["out_neg"]

            vout_p = X[node_index[out_pos]] if out_pos != "0" else 0.0
            vout_n = X[node_index[out_neg]] if out_neg != "0" else 0.0

            vout_diff.append(vout_p - vout_n)

    print("\n===== AC DONE =====")
    print(f"Frequency points : {len(frequencies)}")
    print(f"Frequency range  : {frequencies[0]:.3e} — {frequencies[-1]:.3e} Hz")

    if print_requests:
        print("\n  Peak magnitudes (mid-band estimate):")
        for i, (req_type, node_list) in enumerate(print_requests):
            for j, node in enumerate(node_list):
                vals = np.array(outputs[i][j])
                peak_dB = 20 * np.log10(np.max(np.abs(vals)) + 1e-30)
                print(f"    {req_type.upper()}({node}) peak = {peak_dB:.1f} dB")

    # ── Plot node responses ─────────────────────────
    if print_requests:

        for i, (req_type, node_list) in enumerate(print_requests):

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 5))

            for j, node in enumerate(node_list):

                vals  = np.array(outputs[i][j])
                mag   = 20 * np.log10(np.maximum(np.abs(vals), 1e-30))

                # ✅ FIXED PHASE
                phase = np.angle(vals, deg=True)
                phase = (phase + 180) % 360 - 180

                color = _COLORS[j % len(_COLORS)]
                label = f"{req_type.upper()}({node})"

                ax1.semilogx(frequencies, mag,   color=color,
                             label=label, linewidth=1.4)
                ax2.semilogx(frequencies, phase, color=color,
                             label=label, linewidth=1.4)

            ax1.set_ylabel("Magnitude (dB)")
            ax1.grid(True, which="both", alpha=0.4)
            ax1.ticklabel_format(style='plain', axis='y')
            ax1.yaxis.get_major_formatter().set_useOffset(False)

            ax2.set_ylabel("Phase (deg)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.grid(True, which="both", alpha=0.4)
            ax2.ticklabel_format(style='plain', axis='y')
            ax2.yaxis.get_major_formatter().set_useOffset(False)

            if len(node_list) > 1:
                ax1.legend(fontsize=8, framealpha=0.7)
                ax2.legend(fontsize=8, framealpha=0.7)
            else:
                ax1.set_title(f"AC Response  {req_type.upper()}({node_list[0]})")

            plt.tight_layout()
            plt.show()

    # ── Differential gain ─────────────────────────
    if diff_gain_request is not None:

        vout_diff = np.array(vout_diff)

        gain = vout_diff / (vin_excitation + 1e-30)

        mag   = 20 * np.log10(np.maximum(np.abs(gain), 1e-30))

        # ✅ FIXED PHASE HERE TOO
        phase = np.angle(gain, deg=True)
        phase = (phase + 180) % 360 - 180

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 5))

        ax1.semilogx(frequencies, mag, linewidth=1.6)
        ax1.set_ylabel("Diff Gain (dB)")
        ax1.grid(True, which="both", alpha=0.4)
        ax1.ticklabel_format(style='plain', axis='y')
        ax1.yaxis.get_major_formatter().set_useOffset(False)

        ax2.semilogx(frequencies, phase, linewidth=1.6)
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.grid(True, which="both", alpha=0.4)
        ax2.ticklabel_format(style='plain', axis='y')
        ax2.yaxis.get_major_formatter().set_useOffset(False)

        title = f"Differential Gain: ({diff_gain_request['out_pos']}-{diff_gain_request['out_neg']})"
        plt.suptitle(title)

        plt.tight_layout()
        plt.show()

    return results