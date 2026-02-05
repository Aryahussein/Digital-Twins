import sympy as sp

# ==========================================================
# DC Solver Entry Point
# ==========================================================
def run_dc(NETLIST_FILE):

    # ==========================================================
    # Helper functions
    # ==========================================================
    def strip_comments(line):
        """Remove SPICE comments (* or ;)"""
        for c in ['*', ';']:
            if c in line:
                line = line.split(c, 1)[0]
        return line.strip()

    def parse_value(val):
        """Parse SPICE numeric values with suffixes"""
        val = val.lower()
        scale = {
            't': 1e12,
            'g': 1e9,
            'meg': 1e6,
            'k': 1e3,
            'm': 1e-3,
            'u': 1e-6,
            'n': 1e-9,
            'p': 1e-12,
        }

        for s in scale:
            if val.endswith(s):
                return float(val[:-len(s)]) * scale[s]
        return float(val)

    # ==========================================================
    # Netlist containers
    # ==========================================================
    resistors = []
    currents = []
    voltages = []
    vccs = []
    nodes = set()
    found_op = False

    # ==========================================================
    # Parse netlist
    # ==========================================================
    with open(NETLIST_FILE, "r") as f:
        for raw_line in f:
            line = strip_comments(raw_line)
            if not line:
                continue

            tokens = line.split()
            name = tokens[0].lower()

            if name == '.op':
                found_op = True

            elif name.startswith('r'):
                _, n1, n2, val = tokens
                resistors.append((n1, n2, parse_value(val)))
                nodes.update([n1, n2])

            elif name.startswith('i'):
                _, n_plus, n_minus, val = tokens
                currents.append((n_plus, n_minus, parse_value(val)))
                nodes.update([n_plus, n_minus])

            elif name.startswith('v'):
                _, n_plus, n_minus, val = tokens
                voltages.append((tokens[0], n_plus, n_minus, parse_value(val)))
                nodes.update([n_plus, n_minus])

            elif name.startswith('g'):
                _, n_plus, n_minus, ncp, ncm, val = tokens
                vccs.append((n_plus, n_minus, ncp, ncm, parse_value(val)))
                nodes.update([n_plus, n_minus, ncp, ncm])

    if not found_op:
        raise RuntimeError("DC solver called but no .op directive found")

    # ==========================================================
    # Node indexing
    # ==========================================================
    nodes.discard("0")
    nodes = sorted(nodes)

    N = len(nodes)
    M = len(voltages)

    node_idx = {n: i for i, n in enumerate(nodes)}

    # ==========================================================
    # Symbolic variables
    # ==========================================================
    Vn = sp.Matrix([sp.symbols(f"V{n}") for n in nodes])
    Iv = sp.Matrix([sp.symbols(f"I_{v[0]}") for v in voltages])

    # ==========================================================
    # MNA matrices
    # ==========================================================
    G = sp.zeros(N, N)
    B = sp.zeros(N, M)
    C = sp.zeros(M, N)
    D = sp.zeros(M, M)
    I = sp.zeros(N, 1)
    E = sp.zeros(M, 1)

    # ==========================================================
    # Stamp resistors
    # ==========================================================
    for n1, n2, R in resistors:
        g = 1 / R
        if n1 != "0":
            G[node_idx[n1], node_idx[n1]] += g
        if n2 != "0":
            G[node_idx[n2], node_idx[n2]] += g
        if n1 != "0" and n2 != "0":
            i, j = node_idx[n1], node_idx[n2]
            G[i, j] -= g
            G[j, i] -= g

    # ==========================================================
    # Stamp VCCS
    # ==========================================================
    for n_plus, n_minus, ncp, ncm, gm in vccs:

        def idx(n):
            return node_idx[n] if n != "0" else None

        p, m = idx(n_plus), idx(n_minus)
        cp, cm = idx(ncp), idx(ncm)

        if p is not None and cp is not None:
            G[p, cp] += gm
        if p is not None and cm is not None:
            G[p, cm] -= gm
        if m is not None and cp is not None:
            G[m, cp] -= gm
        if m is not None and cm is not None:
            G[m, cm] += gm

    # ==========================================================
    # Stamp current sources
    # ==========================================================
    for n_plus, n_minus, val in currents:
        if n_plus != "0":
            I[node_idx[n_plus]] -= val
        if n_minus != "0":
            I[node_idx[n_minus]] += val

    # ==========================================================
    # Stamp voltage sources
    # ==========================================================
    for k, (name, n_plus, n_minus, val) in enumerate(voltages):
        if n_plus != "0":
            B[node_idx[n_plus], k] = 1
            C[k, node_idx[n_plus]] = 1
        if n_minus != "0":
            B[node_idx[n_minus], k] = -1
            C[k, node_idx[n_minus]] = -1
        E[k] = val

    # ==========================================================
    # Assemble & solve
    # ==========================================================
    if M > 0:
        A = G.row_join(B)
        A = A.col_join(C.row_join(D))
        Z = I.col_join(E)
        X = Vn.col_join(Iv)
    else:
        A = G
        Z = I
        X = Vn

    X_sol = A.LUsolve(Z)

    # ==========================================================
    # Output
    # ==========================================================
    print("\n========== DC OPERATING POINT ==========")
    for n, v in zip(nodes, X_sol[:N]):
        sp.pprint(sp.Eq(sp.symbols(f"V{n}"), v))

    if M > 0:
        print("\n========== VOLTAGE SOURCE CURRENTS ==========")
        for (name, _, _, _), iv in zip(voltages, X_sol[N:]):
            sp.pprint(sp.Eq(sp.symbols(f"I_{name}"), iv))


# Explicit export
__all__ = ["run_dc"]
