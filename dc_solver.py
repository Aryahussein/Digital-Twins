import numpy as np

GMIN     = 1e-12
MAX_ITER = 150
TOL      = 1e-6


def _stamp_all(components, node_index, N, Mv, Mo, x,
               source_scale=1.0, sweep_overrides=None):
    """
    Assemble MNA matrix G and RHS b.
    Returns (x_new, G, b). x_new is None if the matrix is singular.
    """
    size = N + Mv + Mo

    if size == 0:
        raise RuntimeError(
            "Circuit has zero unknowns (N=Mv=Mo=0). "
            "Check that the netlist file is correct and contains components."
        )

    G = np.zeros((size, size))
    b = np.zeros(size)

    for i in range(size):
        G[i, i] += GMIN

    ctx = {
        "node_index":      node_index,
        "analysis":        "dc",
        "N":               N,
        "Mv":              Mv,
        "x":               x,
        "source_scale":    source_scale,
        "sweep_overrides": sweep_overrides or {},
    }

    for comp in components:
        comp.stamp(G, b, ctx)

    # Source-stepping: scale voltage-source KVL rows after all stamps.
    # VoltageSource.stamp sets b[row] = raw value (no scaling there).
    if source_scale != 1.0:
        for row in range(N, N + Mv):
            b[row] *= source_scale

    try:
        x_new = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return None, G, b

    return x_new, G, b


def _find_vsource(components, name):
    """Return the VoltageSource component with the given name, or None."""
    from MODELS.voltage_source import VoltageSource
    for comp in components:
        if isinstance(comp, VoltageSource) and comp.name.lower() == name.lower():
            return comp
    return None


def _branch_current(comp, x, N):
    """
    Return the MNA branch current (A) for a VoltageSource.
    In MNA the variable x[N + index] is the current flowing FROM n1
    through the source TO n2 (i.e. INTO the + terminal from the circuit).
    Convention: positive = current flows out of + terminal into the circuit
    (same as SPICE: I(Vsrc) > 0 means current flows from + to - through
    the external circuit).
    We negate x[N+index] to match that convention.
    """
    from MODELS.voltage_source import VoltageSource
    if not isinstance(comp, VoltageSource):
        return None
    return -x[N + comp.index]   # negate: MNA stores current INTO the + terminal


def run_dc(components, node_index, N, Mv, Mo,
           sens_node=None, print_requests=None,
           sweep_overrides=None):
    """
    Full DC solve with source stepping + Newton iteration.
    """
    size         = N + Mv + Mo
    source_steps = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    x            = np.zeros(size)

    for step_scale in source_steps:

        print(f"[Source step] scale = {step_scale:.1f}")

        for iteration in range(MAX_ITER):

            x_new, G, b = _stamp_all(
                components, node_index, N, Mv, Mo, x,
                source_scale=step_scale,
                sweep_overrides=sweep_overrides
            )

            if x_new is None:
                raise RuntimeError(
                    f"Singular matrix at source scale={step_scale:.1f}, "
                    f"iter={iteration}. Check for floating nodes."
                )

            err      = np.max(np.abs(x_new - x))
            residual = np.max(np.abs(G @ x_new - b))

            print(f"  [Newton] iter={iteration:3d}  err={err:.3e}  res={residual:.3e}")

            if err < TOL and residual < TOL:
                print(f"  Converged in {iteration+1} iterations.")
                x = x_new
                break

            alpha = 1.0
            if np.max(np.abs((x + (x_new - x))[:N])) > 50.0:
                alpha = 0.5
            x = x + alpha * (x_new - x)

        else:
            raise RuntimeError(
                f"Newton did not converge at source scale={step_scale:.1f}. "
                f"Last err={err:.3e}"
            )

    # Final clean solve at full scale
    x_new, G, b = _stamp_all(
        components, node_index, N, Mv, Mo, x,
        source_scale=1.0, sweep_overrides=sweep_overrides
    )
    if x_new is not None:
        x = x_new

    # ── Operating point summary ───────────────────────────────────────
    print("\n===== DC OPERATING POINT =====")

    print("  Node voltages:")
    for n in node_index:
        print(f"    V({n}) = {x[node_index[n]]:.6f} V")

    # Print branch currents for every voltage source automatically
    from MODELS.voltage_source import VoltageSource
    vsources = [c for c in components if isinstance(c, VoltageSource)]
    if vsources:
        print("  Branch currents:")
        for comp in vsources:
            I = _branch_current(comp, x, N)
            # Choose unit prefix for readability
            if abs(I) >= 1e-3:
                print(f"    I({comp.name}) = {I*1e3:+.6f} mA")
            elif abs(I) >= 1e-6:
                print(f"    I({comp.name}) = {I*1e6:+.6f} µA")
            elif abs(I) >= 1e-9:
                print(f"    I({comp.name}) = {I*1e9:+.6f} nA")
            else:
                print(f"    I({comp.name}) = {I:.6e} A")

    # ── .print requests ───────────────────────────────────────────────
    if print_requests:
        print("\n===== DC PRINT =====")
        for req_type, node_list in print_requests:

            if req_type.lower() == 'v':
                for node in node_list:
                    if node in node_index:
                        print(f"  V({node}) = {x[node_index[node]]:.6f} V")
                    else:
                        print(f"  V({node}) : node not found")

            elif req_type.lower() == 'i':
                # node_list[0] is the voltage source name
                src_name = node_list[0]
                comp = _find_vsource(components, src_name)
                if comp is None:
                    print(f"  I({src_name}) : source not found "
                          f"(only voltage sources have explicit branch currents)")
                else:
                    I = _branch_current(comp, x, N)
                    if abs(I) >= 1e-3:
                        print(f"  I({src_name}) = {I*1e3:+.6f} mA")
                    elif abs(I) >= 1e-6:
                        print(f"  I({src_name}) = {I*1e6:+.6f} µA")
                    elif abs(I) >= 1e-9:
                        print(f"  I({src_name}) = {I*1e9:+.6f} nA")
                    else:
                        print(f"  I({src_name}) = {I:.6e} A")

            else:
                print(f"  Unknown print type: {req_type}")

    # ── Sensitivity ───────────────────────────────────────────────────
    if sens_node is not None:
        print("\n===== DC SENSITIVITY =====")
        x_new, G, b = _stamp_all(
            components, node_index, N, Mv, Mo, x, source_scale=1.0
        )
        c    = np.zeros(size)
        c[node_index[sens_node]] = 1.0
        lam  = np.linalg.solve(G.T, c)
        ctx  = {"node_index": node_index, "x": x}
        for comp in components:
            val = comp.sens_contribution(x, lam, ctx)
            if val is not None:
                print(f"  dV({sens_node})/d{comp.name} = {val:.6e}")

    return x