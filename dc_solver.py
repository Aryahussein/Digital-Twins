import sympy as sp

def run_dc(NETLIST_FILE):

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

    resistors = []
    voltages = []
    currents = []
    opamps = []
    nodes = set()
    found_op = False
    sens_output = None   # <-- NEW

    # ---------------- Parse ----------------
    with open(NETLIST_FILE) as f:
        for raw in f:

            line = strip_comments(raw)
            if not line:
                continue

            tokens = line.split()
            name = tokens[0].lower()

            if name == '.op':
                found_op = True

            elif name == '.sens':   # <-- NEW
                expr = tokens[1].lower()
                if expr.startswith('v(') and expr.endswith(')'):
                    node_name = expr[2:-1]
                    sens_output = ('v', node_name)
                else:
                    raise RuntimeError("Only .sens v(node) supported for now.")

            elif name.startswith('r'):
                _, n1, n2, val = tokens
                resistors.append((tokens[0], n1, n2, parse_value(val)))  # <-- include name
                nodes.update([n1, n2])

            elif name.startswith('v'):
                _, n1, n2, val = tokens
                voltages.append((tokens[0], n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('i'):
                _, n1, n2, val = tokens
                currents.append((n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('o'):
                _, nplus, nminus, nout, gain = tokens
                opamps.append((nplus, nminus, nout, parse_value(gain)))
                nodes.update([nplus, nminus, nout])

    if not found_op:
        raise RuntimeError("No .op directive found")

    nodes.discard("0")
    nodes = sorted(nodes)

    N = len(nodes)
    Mv = len(voltages)
    Mo = len(opamps)

    node_idx = {n: i for i, n in enumerate(nodes)}

    size = N + Mv + Mo

    G = sp.zeros(size, size)
    Z = sp.zeros(size, 1)

    # ---------------- Stamp Resistors ----------------
    for name, n1, n2, R in resistors:
        g = 1 / R
        if n1 != "0":
            G[node_idx[n1], node_idx[n1]] += g
        if n2 != "0":
            G[node_idx[n2], node_idx[n2]] += g
        if n1 != "0" and n2 != "0":
            i, j = node_idx[n1], node_idx[n2]
            G[i, j] -= g
            G[j, i] -= g

    # ---------------- Stamp Current Sources ----------------
    for n1, n2, val in currents:
        if n1 != "0":
            Z[node_idx[n1]] -= val
        if n2 != "0":
            Z[node_idx[n2]] += val

    # ---------------- Stamp Voltage Sources ----------------
    for k, (name, n1, n2, val) in enumerate(voltages):
        row = N + k

        if n1 != "0":
            G[row, node_idx[n1]] = 1
            G[node_idx[n1], row] = 1

        if n2 != "0":
            G[row, node_idx[n2]] = -1
            G[node_idx[n2], row] = -1

        Z[row] = val

    # ---------------- Stamp Ideal Op Amps ----------------
    for k, (nplus, nminus, nout, gain) in enumerate(opamps):

        row = N + Mv + k

        if nout != "0":
            G[row, node_idx[nout]] = 1
            G[node_idx[nout], row] = 1

        if nplus != "0":
            G[row, node_idx[nplus]] -= gain

        if nminus != "0":
            G[row, node_idx[nminus]] += gain

    # ---------------- Solve DC ----------------
    X = G.LUsolve(Z)

    print("\n========== DC OPERATING POINT ==========")
    for n in nodes:
        idx = node_idx[n]
        print(f"V{n} = {float(X[idx])}")

    if Mv > 0:
        print("\n========== VOLTAGE SOURCE CURRENTS ==========")
        for k, (name, _, _, _) in enumerate(voltages):
            print(f"I_{name} = {float(X[N+k])}")

    # ---------------- Adjoint Sensitivity ----------------
    if sens_output is not None:

        print("\n========== ADJOINT SENSITIVITY ==========")

        kind, target = sens_output

        if target == "0":
            raise RuntimeError("Cannot compute sensitivity of ground.")

        if target not in node_idx:
            raise RuntimeError(f"Node {target} not found.")

        c = sp.zeros(size, 1)
        c[node_idx[target]] = 1

        # Solve adjoint system
        lambda_vec = G.T.LUsolve(c)

        for name, n1, n2, R in resistors:

            V1 = X[node_idx[n1]] if n1 != "0" else 0
            V2 = X[node_idx[n2]] if n2 != "0" else 0

            L1 = lambda_vec[node_idx[n1]] if n1 != "0" else 0
            L2 = lambda_vec[node_idx[n2]] if n2 != "0" else 0

            sens = (1 / R**2) * (V1 - V2) * (L1 - L2)

            print(f"dV{target}/d{name} = {float(sens)}")
