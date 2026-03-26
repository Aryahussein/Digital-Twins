import numpy as np

GMIN = 1e-12


def _stamp_all(components, node_index, N, Mv, Mo, x, source_scale=1.0):
    size = N + Mv + Mo
    G    = np.zeros((size, size))
    b    = np.zeros(size)

    for i in range(size):
        G[i, i] += GMIN

    ctx = {
        "node_index":   node_index,
        "analysis":     "dc",
        "N":            N,
        "Mv":           Mv,
        "x":            x,
        "source_scale": source_scale,
    }

    for comp in components:
        comp.stamp(G, b, ctx)

    if source_scale != 1.0:
        for row in range(N, N + Mv):
            b[row] *= source_scale

    try:
        x_new = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return None, G, b

    return x_new, G, b


def run_dc(components, node_index, N, Mv, Mo, sens_node=None, print_requests=None):

    size         = N + Mv + Mo
    max_iters    = 150
    tol          = 1e-6
    source_steps = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    x            = np.zeros(size)

    for step_scale in source_steps:

        print(f"[Source step] scale = {step_scale:.1f}")

        for iteration in range(max_iters):

            x_new, G, b = _stamp_all(
                components, node_index, N, Mv, Mo, x, source_scale=step_scale
            )

            if x_new is None:
                raise RuntimeError(
                    f"Singular matrix at source scale={step_scale:.1f}, "
                    f"iter={iteration}. Check for floating nodes."
                )

            err      = np.max(np.abs(x_new - x))
            residual = np.max(np.abs(G @ x_new - b))

            print(f"  [Newton] iter={iteration:3d}  err={err:.3e}  res={residual:.3e}")

            if err < tol and residual < tol:
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

    x_new, G, b = _stamp_all(components, node_index, N, Mv, Mo, x, source_scale=1.0)
    if x_new is not None:
        x = x_new

    print("\n===== DC OPERATING POINT =====")
    for n in node_index:
        print(f"  V({n}) = {x[node_index[n]]:.6f} V")

    if print_requests:
        print("\n===== DC PRINT =====")
        # print_requests format: (req_type, [node1, node2, ...])
        for req_type, node_list in print_requests:
            for node in node_list:
                if req_type.lower() == 'v':
                    print(f"  V({node}) = {x[node_index[node]]:.6f} V")

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