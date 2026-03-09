import logging
import numpy as np
from scipy.sparse.linalg import spsolve, splu
from assembleYmatrix import stamp_nonlinear_components

logger = logging.getLogger(__name__)


def solve_sparse(G, I):
    return spsolve(G, I)


def solve_LU(G):
    return splu(G)


def get_node_and_branch_currents(lu, sources):
    return lu.solve(sources)


def solve_linear_circuit(Y, sources):
    lu = solve_LU(Y)
    VI = get_node_and_branch_currents(lu, sources)
    return lu, VI


def solve_adjoint(lu, target, node_map):
    d = np.zeros(len(node_map))
    idx = node_map[target]
    d[idx] = 1.0
    return lu.solve(d, trans="T")


def solve_nonlinear_circuit(
    Y_base,
    sources_base,
    components,
    node_map,
    V_ini,
    max_iter=200,
    tol=1e-6,
    num_steps=10,
    print_stuff=False,
    gmin=1e-12,
):
    """
    Newton-Raphson solver with GMIN stepping for convergence.

    Strategy (mirrors real SPICE):
    1. First attempt: solve directly with small GMIN.
    2. If that fails: GMIN stepping — start with large GMIN (1e-3) and
       gradually reduce to the target GMIN. This keeps floating nodes
       grounded during early iterations.
    3. Source ramping is NOT used because it breaks circuits where devices
       need full supply voltage to turn on (e.g., CMOS inverters).
    """
    from scipy.sparse import diags as sp_diags

    n = len(node_map)

    def _build_gmin_matrix(gmin_val):
        gmin_vec = np.zeros(n)
        for key, idx in node_map.items():
            if isinstance(key, int):
                gmin_vec[idx] = gmin_val
        return sp_diags(gmin_vec, 0, shape=(n, n), format="csc")

    def _newton_solve(V_start, Y_gmin_mat, label=""):
        """Run Newton-Raphson iterations with a given GMIN matrix."""
        V_k = V_start.copy()
        prev_V_k = V_start.copy()
        lu = None
        prev_error = 1e30

        for comp in components.values():
            if comp.get("type") == "O":
                comp["_gain_scale"] = 1.0

        for i in range(max_iter):
            Y_iter = Y_base.copy()
            sources_iter = sources_base.copy()

            Y_iter, sources_iter = stamp_nonlinear_components(
                Y_iter, sources_iter, components, node_map, prev_V_k, V_k
            )

            # Add GMIN
            Y_iter = Y_iter + Y_gmin_mat

            try:
                lu, V_new = solve_linear_circuit(Y_iter, sources_iter)
            except RuntimeError:
                return None, V_k, False

            max_error = np.max(np.abs(V_new - V_k))
            prev_V_k = V_k.copy()

            # Full Newton step — pnjlim inside the device stamps
            # handles junction voltage limiting, which is the standard
            # SPICE approach. External damping interferes with pnjlim.
            V_k = V_new.copy()

            logger.debug("  %s iter %d, error=%.2e", label, i, max_error)

            if max_error < tol:
                logger.debug("  %s converged in %d iterations.", label, i + 1)
                return lu, V_k, True

            # Detect divergence: if error grows 100x, give up early
            if max_error > 1e15:
                return None, V_k, False

        return None, V_k, False

    # --- Attempt 1: Direct solve with target GMIN ---
    Y_gmin = _build_gmin_matrix(gmin)
    lu, V_k, converged = _newton_solve(V_ini, Y_gmin, label="Direct")
    if converged:
        return lu, V_k

    # --- Attempt 2: GMIN stepping (from zero initial guess) ---
    logger.info("Direct solve failed. Starting GMIN stepping...")
    gmin_schedule = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, gmin]

    V_k = np.zeros(n)  # GMIN stepping works best from zero
    gmin_ok = True
    for step_gmin in gmin_schedule:
        Y_gmin_step = _build_gmin_matrix(step_gmin)
        lu, V_k, converged = _newton_solve(V_k, Y_gmin_step, label=f"GMIN={step_gmin:.0e}")
        if not converged:
            gmin_ok = False
            break

    if gmin_ok:
        for comp in components.values():
            if comp.get("type") == "O":
                comp["_gain_scale"] = 1.0
        return lu, V_k

    # --- Attempt 3: Source ramping with small GMIN ---
    # This works better for BJT circuits where GMIN stepping fails
    # because large GMIN drowns out junction conductances.
    logger.info("GMIN stepping failed. Trying source ramping...")
    Y_gmin_small = _build_gmin_matrix(gmin)
    ramp_steps = np.linspace(0.1, 1.0, 10)
    V_k = V_ini.copy()

    for k in ramp_steps:
        sources_ramped = sources_base.copy() * k

        def _newton_ramp(V_start, label=""):
            V_kk = V_start.copy()
            prev_V_kk = V_start.copy()
            lu_r = None
            for it in range(max_iter):
                Y_it = Y_base.copy()
                src_it = sources_ramped.copy()
                Y_it, src_it = stamp_nonlinear_components(
                    Y_it, src_it, components, node_map, prev_V_kk, V_kk
                )
                Y_it = Y_it + Y_gmin_small
                try:
                    lu_r, V_new = solve_linear_circuit(Y_it, src_it)
                except RuntimeError:
                    return None, V_kk, False

                delta = np.abs(V_new - V_kk)
                err = np.max(delta)
                prev_V_kk = V_kk.copy()
                alpha = 1.0 if err <= 1.0 else min(1.0, 1.0 / err)
                V_kk = V_kk + alpha * (V_new - V_kk)
                if err < tol:
                    return lu_r, V_kk, True
            return None, V_kk, False

        lu, V_k, converged = _newton_ramp(V_k, label=f"Ramp={k:.1f}")
        if not converged:
            raise RuntimeError(
                f"Newton-Raphson failed during source ramping at {k*100:.0f}%."
            )

    # Reset opamp scaling
    for comp in components.values():
        if comp.get("type") == "O":
            comp["_gain_scale"] = 1.0

    return lu, V_k
