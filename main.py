import sympy as sp

NETLIST_FILE = "test_circuit.sp"

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
resistors = []      # (n1, n2, R)
currents = []       # (n+, n-, I)
voltages = []       # (name, n+, n-, V)
vccs = []   # (n+, n-, nc+, nc-, gm)
nodes = set()

# ==========================================================
# Parse netlist
# ==========================================================
with open(NETLIST_FILE, "r") as f:
    for raw_line in f:
        line = strip_comments(raw_line)
        if not line:
            continue

        tokens = line.split()
        name = tokens[0]

        if name[0].upper() == 'R':
            _, n1, n2, val = tokens
            resistors.append((n1, n2, parse_value(val)))
            nodes.update([n1, n2])

        elif name[0].upper() == 'I':
            _, np, nm, val = tokens
            currents.append((np, nm, parse_value(val)))
            nodes.update([np, nm])

        elif name[0].upper() == 'V':
            _, np, nm, val = tokens
            voltages.append((name, np, nm, parse_value(val)))
            nodes.update([np, nm])
        elif name[0].upper() == 'G':
            _, np, nm, ncp, ncm, val = tokens
            vccs.append((np, nm, ncp, ncm, parse_value(val)))
            nodes.update([np, nm, ncp, ncm])


# Ground handling
nodes.discard("0")
nodes = sorted(nodes)

N = len(nodes)          # number of nodes
M = len(voltages)       # number of voltage sources

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

# ==========================================================
# Stamp VCCS (Voltage Controlled Current Sources)
# ==========================================================
for np, nm, ncp, ncm, gm in vccs:

    def idx(n):
        return node_idx[n] if n != "0" else None

    p  = idx(np)
    m  = idx(nm)
    cp = idx(ncp)
    cm = idx(ncm)

    # Contribution to node n+
    if p is not None and cp is not None:
        G[p, cp] += gm
    if p is not None and cm is not None:
        G[p, cm] -= gm

    # Contribution to node n-
    if m is not None and cp is not None:
        G[m, cp] -= gm
    if m is not None and cm is not None:
        G[m, cm] += gm


# ==========================================================
# Stamp current sources
# ==========================================================
for np, nm, val in currents:
    if np != "0":
        I[node_idx[np]] -= val
    if nm != "0":
        I[node_idx[nm]] += val

# ==========================================================
# Stamp voltage sources
# ==========================================================
for k, (name, np, nm, val) in enumerate(voltages):
    if np != "0":
        B[node_idx[np], k] = 1
        C[k, node_idx[np]] = 1
    if nm != "0":
        B[node_idx[nm], k] = -1
        C[k, node_idx[nm]] = -1
    E[k] = val

# ==========================================================
# Assemble MNA system (handle M = 0 correctly)
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

# ==========================================================
# Output
# ==========================================================
print("\n================= UNKNOWN VECTOR =================")
sp.pprint(X)

print("\n================= SYSTEM MATRIX A =================")
sp.pprint(A)

print("\n================= RHS VECTOR Z ===================")
sp.pprint(Z)

print("\n============= NODAL EQUATIONS ====================")
eqs = A * X - Z
for eq in eqs:
    sp.pprint(sp.Eq(eq, 0))


# ==========================================================
# Nodal Admittance Matrix
# ==========================================================
Y = G.copy()

print("\n=========== NODAL ADMITTANCE MATRIX (Y) ===========")
sp.pprint(Y)

print("\n=========== NODE VOLTAGE VECTOR (V) ===============")
sp.pprint(Vn)

print("\n=========== CURRENT INJECTION VECTOR (I) ==========")
sp.pprint(I)

print("\n=========== NODAL EQUATIONS Y·V = I ===============")
for i in range(N):
    expr = sum(Y[i, j] * Vn[j] for j in range(N))
    sp.pprint(sp.Eq(expr, I[i]))

# ==========================================================
# Solve Y·V = I using LU (only valid if no voltage sources)
# ==========================================================
if M == 0:
    print("\n=========== SOLVING Y·V = I USING LU ===========")

    try:
        # LU decomposition
        L, U, perm = Y.LUdecomposition()

        print("\nL matrix:")
        sp.pprint(L)

        print("\nU matrix:")
        sp.pprint(U)

        # Solve using LU
        V_sol = Y.LUsolve(I)

        print("\n=========== NODE VOLTAGE SOLUTION ===========")
        for n, v in zip(nodes, V_sol):
            sp.pprint(sp.Eq(sp.symbols(f"V{n}"), v))

    except Exception as e:
        print("\nLU solve failed:")
        print(e)

else:
    # ==========================================================
    # Solve full MNA system A·X = Z using LU (voltage sources)
    # ==========================================================
    if M > 0:
        print("\n=========== SOLVING FULL MNA SYSTEM A·X = Z USING LU ===========")

        try:
            # LU decomposition of full MNA matrix
            L, U, perm = A.LUdecomposition()

            print("\nL matrix:")
            sp.pprint(L)

            print("\nU matrix:")
            sp.pprint(U)

            # Solve system
            X_sol = A.LUsolve(Z)

            print("\n=========== SOLUTION VECTOR X ===========")
            sp.pprint(X_sol)

            # --------------------------------------------------
            # Extract node voltages
            # --------------------------------------------------
            print("\n=========== NODE VOLTAGE SOLUTION ===========")
            for n, v in zip(nodes, X_sol[:N]):
                sp.pprint(sp.Eq(sp.symbols(f"V{n}"), v))

            # --------------------------------------------------
            # Extract voltage source currents
            # --------------------------------------------------
            print("\n=========== VOLTAGE SOURCE CURRENTS ===========")
            for (name, _, _, _), iv in zip(voltages, X_sol[N:]):
                sp.pprint(sp.Eq(sp.symbols(f"I_{name}"), iv))

        except Exception as e:
            print("\nMNA LU solve failed:")
            print(e)

