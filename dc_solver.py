import numpy as np


def run_dc(components, node_index, N, Mv, Mo, sens_node=None, print_requests=None):

    size = N + Mv + Mo

    max_iters = 100
    tol = 1e-6
    damping = 1.0

    x = np.zeros(size)

    for iteration in range(max_iters):

        G = np.zeros((size, size))
        b = np.zeros(size)

        ctx = {
            "node_index": node_index,
            "analysis": "dc",
            "N": N,
            "Mv": Mv,
            "x": x
        }

        for comp in components:
            comp.stamp(G, b, ctx)

        try:
            x_new = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            raise RuntimeError("Matrix is singular")

        err = np.max(np.abs(x_new - x))
        residual = np.max(np.abs(G @ x_new - b))

        print(f"[Newton] Iter {iteration}: err={err}, res={residual}")

        if err < tol and residual < tol:
            print(f"[Newton] Converged in {iteration} iterations\n")
            x = x_new
            break

        # -------- Adaptive damping --------
        alpha = damping
        x_trial = x + alpha * (x_new - x)

        if np.max(np.abs(x_trial)) > 1e3:
            alpha *= 0.5

        x = x + alpha * (x_new - x)

    else:
        raise RuntimeError("Newton did not converge")

    print("===== DC OPERATING POINT =====")
    for n in node_index:
        print(f"V({n}) = {x[node_index[n]]}")

    if print_requests:
        print("\n===== DC PRINT =====")
        for req_type, node in print_requests:
            if req_type.lower() == 'v':
                print(f"V({node}) = {x[node_index[node]]}")

    if sens_node is not None:

        print("\n===== DC SENSITIVITY =====")

        G = np.zeros((size, size))
        b = np.zeros(size)

        ctx["x"] = x

        for comp in components:
            comp.stamp(G, b, ctx)

        c = np.zeros(size)
        c[node_index[sens_node]] = 1

        lam = np.linalg.solve(G.T, c)

        for comp in components:
            val = comp.sens_contribution(x, lam, ctx)
            if val is not None:
                print(f"dV({sens_node})/d{comp.name} = {val}")

    return x