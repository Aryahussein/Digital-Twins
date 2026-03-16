import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def strip_comments(line):
    for c in ['*', ';']:
        if c in line:
            line = line.split(c, 1)[0]
    return line.strip()


def parse_value(val):
    val = val.lower()

    scale = {
        'meg': 1e6,
        't': 1e12,
        'g': 1e9,
        'k': 1e3,
        'm': 1e-3,
        'u': 1e-6,
        'n': 1e-9,
        'p': 1e-12
    }

    for s in sorted(scale.keys(), key=len, reverse=True):
        if val.endswith(s):
            return float(val[:-len(s)]) * scale[s]

    return float(val)


def generate_ac_frequencies(ac_sweep):

    sweep_type, npts, fstart, fstop = ac_sweep

    if sweep_type == 'dec':
        decades = np.log10(fstop / fstart)
        total_pts = int(npts * decades)
        return np.logspace(np.log10(fstart), np.log10(fstop), total_pts)

    elif sweep_type == 'oct':
        octaves = np.log2(fstop / fstart)
        total_pts = int(npts * octaves)
        return np.logspace(np.log10(fstart), np.log10(fstop), total_pts)

    elif sweep_type == 'lin':
        return np.linspace(fstart, fstop, npts)

    else:
        raise RuntimeError("Unsupported AC sweep type")


def run_ac(netlist_file):

    resistors = []
    capacitors = []
    inductors = []
    voltages = []
    currents = []
    opamps = []
    nodes = set()

    ac_sweep = None
    sens_output = None

    # -------- Parse Netlist --------
    with open(netlist_file) as f:

        for raw in f:

            line = strip_comments(raw)
            if not line:
                continue

            t = line.split()
            name = t[0].lower()

            if name.startswith('r'):
                _, n1, n2, val = t
                resistors.append((t[0], n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('c'):
                _, n1, n2, val = t
                capacitors.append((n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('l'):
                _, n1, n2, val = t
                inductors.append((n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('v'):

                if len(t) == 5 and t[3].lower() == "ac":
                    _, n1, n2, _, val = t
                else:
                    _, n1, n2, val = t

                voltages.append((t[0], n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('i'):
                _, n1, n2, val = t
                currents.append((n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('o'):
                _, nplus, nminus, nout, gain = t
                opamps.append((nplus, nminus, nout, parse_value(gain)))
                nodes.update([nplus, nminus, nout])

            elif name == '.ac':
                _, sweep_type, npts, fstart, fstop = t
                ac_sweep = (
                    sweep_type.lower(),
                    int(npts),
                    parse_value(fstart),
                    parse_value(fstop)
                )

            elif name == '.sens':

                expr = t[1].lower()

                if expr.startswith("v(") and expr.endswith(")"):
                    sens_output = expr[2:-1]

    if ac_sweep is None:
        raise RuntimeError("No .ac directive found")

    nodes.discard("0")
    nodes = sorted(nodes)

    node_idx = {n: i for i, n in enumerate(nodes)}

    N = len(nodes)
    Mv = len(voltages)
    Mo = len(opamps)

    size = N + Mv + Mo

    frequencies = generate_ac_frequencies(ac_sweep)

    out_node = sens_output if sens_output else nodes[-1]
    out_idx = node_idx[out_node]

    mag = []
    phase = []

    sens_results = {r[0]: [] for r in resistors} if sens_output else None

    # -------- Frequency Sweep --------
    for f in frequencies:

        w = 2 * np.pi * f
        jw = 1j * w

        G = np.zeros((size, size), dtype=complex)
        Z = np.zeros(size, dtype=complex)

        # Resistors
        for name, n1, n2, R in resistors:

            g = 1 / R

            if n1 != "0":
                i = node_idx[n1]
                G[i, i] += g

            if n2 != "0":
                j = node_idx[n2]
                G[j, j] += g

            if n1 != "0" and n2 != "0":
                i = node_idx[n1]
                j = node_idx[n2]
                G[i, j] -= g
                G[j, i] -= g

        # Capacitors
        for n1, n2, C in capacitors:

            yc = jw * C

            if n1 != "0":
                G[node_idx[n1], node_idx[n1]] += yc

            if n2 != "0":
                G[node_idx[n2], node_idx[n2]] += yc

            if n1 != "0" and n2 != "0":
                i = node_idx[n1]
                j = node_idx[n2]
                G[i, j] -= yc
                G[j, i] -= yc

        # Inductors
        for n1, n2, L in inductors:

            yl = 1 / (jw * L)

            if n1 != "0":
                G[node_idx[n1], node_idx[n1]] += yl

            if n2 != "0":
                G[node_idx[n2], node_idx[n2]] += yl

            if n1 != "0" and n2 != "0":
                i = node_idx[n1]
                j = node_idx[n2]
                G[i, j] -= yl
                G[j, i] -= yl

        # Current sources
        for n1, n2, val in currents:

            if n1 != "0":
                Z[node_idx[n1]] -= val

            if n2 != "0":
                Z[node_idx[n2]] += val

        # Voltage sources
        for k, (name, n1, n2, val) in enumerate(voltages):

            row = N + k

            if n1 != "0":
                G[row, node_idx[n1]] = 1
                G[node_idx[n1], row] = 1

            if n2 != "0":
                G[row, node_idx[n2]] = -1
                G[node_idx[n2], row] = -1

            Z[row] = val

        # Solve
        X = np.linalg.solve(G, Z)

        V = X[:N]
        vout = V[out_idx]

        print(f, vout)

        mag.append(abs(vout))
        phase.append(np.angle(vout, deg=True))

        # -------- Adjoint Sensitivity --------
        if sens_output:

            c = np.zeros(size, dtype=complex)
            c[out_idx] = 1

            lam = np.linalg.solve(G.conj().T, c)
            lam = lam[:N]

            for name, n1, n2, R in resistors:

                V1 = V[node_idx[n1]] if n1 != "0" else 0
                V2 = V[node_idx[n2]] if n2 != "0" else 0

                L1 = lam[node_idx[n1]] if n1 != "0" else 0
                L2 = lam[node_idx[n2]] if n2 != "0" else 0

                sens = (1/R**2) * (V1 - V2) * (L1 - L2)

                sens_results[name].append(sens)

    # -------- Plotting --------
    mag = np.array(mag)
    phase = np.array(phase)
    frequencies = np.array(frequencies)

    mag_db = 20*np.log10(np.maximum(mag, 1e-30))

    print("AC sweep points:", len(frequencies))

    fig, (ax_mag, ax_phase) = plt.subplots(2,1,sharex=True,figsize=(7,6))

    ax_mag.semilogx(frequencies, mag_db)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title(f"Bode Plot (Node {out_node})")
    ax_mag.grid(True, which="both")

    ax_phase.semilogx(frequencies, phase)
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, which="both")

    plt.tight_layout()
    plt.show()

    # -------- Sensitivity Plot --------
    if sens_output is not None:

        for name in sens_results:

            sens_vals = np.array(sens_results[name])

            sens_mag = 20*np.log10(np.maximum(np.abs(sens_vals), 1e-30))
            sens_phase = np.angle(sens_vals, deg=True)

            fig, (ax_smag, ax_sphase) = plt.subplots(2,1,sharex=True,figsize=(7,6))

            ax_smag.semilogx(frequencies, sens_mag)
            ax_smag.set_ylabel("|dV/dR| (dB)")
            ax_smag.set_title(f"Sensitivity of V({out_node}) w.r.t {name}")
            ax_smag.grid(True, which="both")

            ax_sphase.semilogx(frequencies, sens_phase)
            ax_sphase.set_xlabel("Frequency (Hz)")
            ax_sphase.set_ylabel("Phase (deg)")
            ax_sphase.grid(True, which="both")

            plt.tight_layout()
            plt.show()