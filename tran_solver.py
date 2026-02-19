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
# Transient Solver
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
                # Oname n+ n- nout gain
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

    # Assign voltage source indices
    vs_indices = {}
    for k in range(m):
        name, n1, n2, stype, params = voltage_sources[k]
        row = n + k
        voltage_sources[k] = (name, n1, n2, stype, params, row)
        vs_indices[name] = row

    # Assign inductor indices
    ind_indices = {}
    for k in range(l):
        name, n1, n2, value = inductors[k]
        row = n + m + k
        inductors[k] = (name, n1, n2, value, row)
        ind_indices[name] = row

    # Assign op amp indices
    op_indices = {}
    for k in range(o):
        name, nplus, nminus, nout, gain = opamps[k]
        row = n + m + l + k
        opamps[k] = (name, nplus, nminus, nout, gain, row)
        op_indices[name] = row

    x_prev = np.zeros(size)

    time_vec = []
    outputs = [[] for _ in print_requests]

    steps = int(tstop / dt)

    # ============================================================
    # Time Loop
    # ============================================================

    for step in range(steps + 1):

        t = step * dt

        G = np.zeros((size, size))
        b = np.zeros(size)

        # ---- Resistors ----
        for name, n1, n2, value in resistors:
            g = 1.0 / value
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

        # ---- Capacitors ----
        for name, n1, n2, value in capacitors:
            g = value / dt
            v_prev = 0

            if n1 != 0:
                i = node_index[n1]
                v_prev += x_prev[i]
                G[i, i] += g
            if n2 != 0:
                j = node_index[n2]
                v_prev -= x_prev[j]
                G[j, j] += g
            if n1 != 0 and n2 != 0:
                i, j = node_index[n1], node_index[n2]
                G[i, j] -= g
                G[j, i] -= g

            Ieq = g * v_prev

            if n1 != 0:
                b[node_index[n1]] += Ieq
            if n2 != 0:
                b[node_index[n2]] -= Ieq

        # ---- Inductors ----
        for name, n1, n2, value, row in inductors:

            Req = value / dt
            Iprev = x_prev[row]

            if n1 != 0:
                i = node_index[n1]
                G[i, row] += 1
                G[row, i] += 1
            if n2 != 0:
                j = node_index[n2]
                G[j, row] -= 1
                G[row, j] -= 1

            G[row, row] -= Req
            b[row] += Req * Iprev

        # ---- Voltage Sources ----
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

        # ---- Ideal Op Amps ----
        for name, nplus, nminus, nout, gain, row in opamps:

            if nout != 0:
                G[row, node_index[nout]] = 1
                G[node_index[nout], row] = 1

            if nplus != 0:
                G[row, node_index[nplus]] -= gain

            if nminus != 0:
                G[row, node_index[nminus]] += gain

        x = np.linalg.solve(G, b)

        # ---- Store outputs ----
        for idx, (req_type, req_target) in enumerate(print_requests):

            target = req_target.upper()

            if req_type == "v":
                outputs[idx].append(x[node_index[int(target)]])

            elif req_type == "i":
                if target in vs_indices:
                    outputs[idx].append(x[vs_indices[target]])
                elif target in ind_indices:
                    outputs[idx].append(x[ind_indices[target]])
                elif target in op_indices:
                    outputs[idx].append(x[op_indices[target]])

        time_vec.append(t)
        x_prev = x.copy()

    # ---- Subplots ----
    fig, axes = plt.subplots(len(print_requests), 1, sharex=True,
                             figsize=(7, 3 * len(print_requests)))

    if len(print_requests) == 1:
        axes = [axes]

    for idx, (req_type, req_target) in enumerate(print_requests):
        axes[idx].plot(time_vec, outputs[idx])
        axes[idx].set_ylabel(f"{req_type} {req_target}")
        axes[idx].grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Transient Response")
    plt.tight_layout()
    plt.show()
