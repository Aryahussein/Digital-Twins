import numpy as np
import math
import re
import matplotlib.pyplot as plt


# ============================================================
# Utility
# ============================================================

def parse_value(val):

    multipliers = {
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        'k': 1e3,
        'meg': 1e6,
        'g': 1e9
    }

    val = val.lower()

    for key in multipliers:
        if val.endswith(key):
            return float(val[:-len(key)]) * multipliers[key]

    return float(val)


# ============================================================
# Source Evaluation
# ============================================================

def evaluate_source(source_type, params, t):

    if source_type == "STEP":
        v1, v2, tdelay = params
        return v1 if t < tdelay else v2

    if source_type == "RAMP":
        v1, v2, tstart, trise = params
        if t < tstart:
            return v1
        if t > tstart + trise:
            return v2
        slope = (v2 - v1) / trise
        return v1 + slope * (t - tstart)

    if source_type == "SINE":
        voff, vamp, freq = params
        return voff + vamp * math.sin(2 * math.pi * freq * t)

    if source_type == "PULSE":
        vlow, vhigh, tdelay, trise, tfall, ton, period = params
        if t < tdelay:
            return vlow
        local = (t - tdelay) % period
        return vhigh if local < ton else vlow

    return 0


# ============================================================
# Transient Solver with Sensitivity
# ============================================================

def run_tran(netlist_file):

    resistors = []
    capacitors = []
    inductors = []
    voltage_sources = []
    opamps = []

    nodes = set()

    dt = None
    tstop = None

    print_requests = []
    sens_node = None

    # ---------------- Parse ----------------

    with open(netlist_file) as f:

        for raw in f:

            line = raw.strip()

            if not line or line.startswith("*"):
                continue

            tokens = line.split()
            keyword = tokens[0].lower()

            if keyword == ".tran":
                dt = parse_value(tokens[1])
                tstop = parse_value(tokens[2])
                continue

            if keyword == ".print":
                print_requests.append((tokens[1].lower(), tokens[2]))
                continue

            if keyword == ".sens":

                expr = tokens[1].lower()

                if expr.startswith("v(") and expr.endswith(")"):
                    sens_node = int(expr[2:-1])

                continue

            if keyword == ".end":
                break

            element_type = tokens[0][0].upper()
            name = tokens[0]

            if element_type == "R":
                n1, n2 = int(tokens[1]), int(tokens[2])
                value = parse_value(tokens[3])
                resistors.append((name.upper(), n1, n2, value))
                nodes.update([n1, n2])

            elif element_type == "C":
                n1, n2 = int(tokens[1]), int(tokens[2])
                value = parse_value(tokens[3])
                capacitors.append((name.upper(), n1, n2, value))
                nodes.update([n1, n2])

            elif element_type == "L":
                n1, n2 = int(tokens[1]), int(tokens[2])
                value = parse_value(tokens[3])
                inductors.append((name.upper(), n1, n2, value))
                nodes.update([n1, n2])

            elif element_type == "V":

                n1, n2 = int(tokens[1]), int(tokens[2])

                match = re.search(r'(\w+)\((.*?)\)', line)

                source_type = match.group(1).upper()
                params = [parse_value(p) for p in match.group(2).split()]

                voltage_sources.append((name.upper(), n1, n2, source_type, params))
                nodes.update([n1, n2])

            elif element_type == "O":

                nplus = int(tokens[1])
                nminus = int(tokens[2])
                nout = int(tokens[3])
                gain = parse_value(tokens[4])

                opamps.append((name.upper(), nplus, nminus, nout, gain))
                nodes.update([nplus, nminus, nout])

    if dt is None or tstop is None:
        raise RuntimeError("No .tran statement found")

    nodes.discard(0)
    nodes = sorted(nodes)

    n = len(nodes)
    m = len(voltage_sources)
    l = len(inductors)
    o = len(opamps)

    node_index = {node: i for i, node in enumerate(nodes)}

    size = n + m + l + o

    # Index assignment
    for k in range(m):
        name, n1, n2, stype, params = voltage_sources[k]
        row = n + k
        voltage_sources[k] = (name, n1, n2, stype, params, row)

    for k in range(l):
        name, n1, n2, value = inductors[k]
        row = n + m + k
        inductors[k] = (name, n1, n2, value, row)

    for k in range(o):
        name, nplus, nminus, nout, gain = opamps[k]
        row = n + m + l + k
        opamps[k] = (name, nplus, nminus, nout, gain, row)

    x_prev = np.zeros(size)

    steps = int(tstop / dt)

    time_vec = []
    x_history = []

    outputs = [[] for _ in print_requests]

    # ============================================================
    # Forward transient simulation
    # ============================================================

    for step in range(steps + 1):

        t = step * dt

        G = np.zeros((size, size))
        b = np.zeros(size)

        # Resistors
        for name, n1, n2, value in resistors:

            g = 1 / value

            if n1 != 0:
                i = node_index[n1]
                G[i, i] += g

            if n2 != 0:
                j = node_index[n2]
                G[j, j] += g

            if n1 != 0 and n2 != 0:
                i, j = node_index[n1], node_index[n2]
                G[i, j] -= g
                G[j, i] -= g

        # Capacitors (BE)
        for name, n1, n2, value in capacitors:

            g = value / dt

            v_prev = 0

            if n1 != 0:
                v_prev += x_prev[node_index[n1]]
                G[node_index[n1], node_index[n1]] += g

            if n2 != 0:
                v_prev -= x_prev[node_index[n2]]
                G[node_index[n2], node_index[n2]] += g

            if n1 != 0 and n2 != 0:
                i, j = node_index[n1], node_index[n2]
                G[i, j] -= g
                G[j, i] -= g

            Ieq = g * v_prev

            if n1 != 0:
                b[node_index[n1]] += Ieq
            if n2 != 0:
                b[node_index[n2]] -= Ieq

        # Voltage sources
        for name, n1, n2, stype, params, row in voltage_sources:

            v = evaluate_source(stype, params, t)

            if n1 != 0:
                i = node_index[n1]
                G[row, i] = 1
                G[i, row] = 1

            if n2 != 0:
                j = node_index[n2]
                G[row, j] = -1
                G[j, row] = -1

            b[row] = v

        x = np.linalg.solve(G, b)

        x_history.append(x.copy())
        x_prev = x.copy()

        time_vec.append(t)

    # ============================================================
    # Adjoint sensitivity
    # ============================================================

    if sens_node is not None:

        print("\n========== TRANSIENT SENSITIVITY ==========")

        lam = np.zeros(size)

        lam[node_index[sens_node]] = 1

        sens = {r[0]: 0.0 for r in resistors}

        for k in reversed(range(len(x_history))):

            xk = x_history[k]

            for name, n1, n2, value in resistors:

                gprime = -1 / (value**2)

                v1 = xk[node_index[n1]] if n1 != 0 else 0
                v2 = xk[node_index[n2]] if n2 != 0 else 0

                dv = v1 - v2

                lam1 = lam[node_index[n1]] if n1 != 0 else 0
                lam2 = lam[node_index[n2]] if n2 != 0 else 0

                sens[name] += (lam1 - lam2) * gprime * dv

        for name in sens:
            print(f"dV({sens_node})/d{name} = {sens[name]}")

    # ============================================================
    # Plot outputs
    # ============================================================

    if print_requests:

        fig, axes = plt.subplots(len(print_requests), 1, sharex=True,
                                 figsize=(7, 3 * len(print_requests)))

        if len(print_requests) == 1:
            axes = [axes]

        for idx, (req_type, req_target) in enumerate(print_requests):

            target = int(req_target)

            vals = [x[node_index[target]] for x in x_history]

            axes[idx].plot(time_vec, vals)
            axes[idx].set_ylabel(f"{req_type} {req_target}")
            axes[idx].grid(True)

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()