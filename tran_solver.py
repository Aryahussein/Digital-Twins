import numpy as np
import matplotlib.pyplot as plt


def run_tran(components, node_index, N, Mv, Mo,
             dt, tstop,
             sens_node=None,
             print_requests=None):

    size = N + Mv + Mo
    steps = int(tstop / dt)

    # ================================
    # Storage
    # ================================
    x_prev = np.zeros(size)
    history = []
    time_vec = []

    outputs = [[] for _ in print_requests] if print_requests else None

    # ================================
    # Newton Parameters
    # ================================
    max_iters = 50
    tol = 1e-6
    damping = 1.0

    # ================================
    # Time stepping
    # ================================
    for step in range(steps + 1):

        t = step * dt

        # Initial guess = previous solution
        x = x_prev.copy()

        # ================================
        # Newton loop
        # ================================
        for iteration in range(max_iters):

            G = np.zeros((size, size))
            b = np.zeros(size)

            ctx = {
                "node_index": node_index,
                "analysis": "tran",
                "dt": dt,
                "x_prev": x_prev,   # BE memory
                "x": x,             # Newton state
                "N": N,
                "Mv": Mv,
                "t": t
            }

            for comp in components:
                comp.stamp(G, b, ctx)

            try:
                x_new = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                raise RuntimeError("Transient matrix singular")

            err = np.max(np.abs(x_new - x))

            if iteration == 0 and step % 50 == 0:
                print(f"[t={t:.3e}] starting Newton")

            if err < tol:
                break

            # Damping
            x = x + damping * (x_new - x)

        else:
            raise RuntimeError(f"Newton failed at t={t}")

        # Converged solution
        x = x_new

        history.append(x.copy())
        x_prev = x.copy()
        time_vec.append(t)

        # ================================
        # Print outputs
        # ================================
        if print_requests:
            for i, (_, node) in enumerate(print_requests):
                outputs[i].append(x[node_index[node]])

    print("\n===== TRANSIENT DONE =====")
    print("Steps:", len(history))

    # ================================
    # Plot
    # ================================
    if print_requests:

        fig, axes = plt.subplots(len(print_requests), 1, sharex=True)

        if len(print_requests) == 1:
            axes = [axes]

        for i, (req_type, node) in enumerate(print_requests):

            axes[i].plot(time_vec, outputs[i])
            axes[i].set_ylabel(f"{req_type}({node})")
            axes[i].grid(True)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    return history