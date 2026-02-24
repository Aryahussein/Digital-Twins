from scipy.sparse.linalg import spsolve, splu
import numpy as np
from assembleYmatrix import stamp_nonlinear_components
from constants import *


def solve_sparse(G, I):
    return spsolve(G, I)


def solve_LU(G):
    return splu(G)


def get_node_and_branch_currents(lu, sources):
    return lu.solve(sources)


def solve_linear_circuit(Y, sources):
    if hasattr(Y, "tocsc"):
        Y = Y.tocsc()
    lu = solve_LU(Y)
    VI = get_node_and_branch_currents(lu, sources)
    return lu, VI


def solve_adjoint(lu, target, node_map):
    d = np.zeros(len(node_map))
    idx = node_map[target]
    d[idx] = 1.0
    return lu.solve(d, trans='T')


def _has_opamps(components):
    return any(str(c.get("type", "")).upper() == "A" for c in components.values())


def _max_opamp_k(components, default=1e3):
    ks = [float(c.get("k", default)) for c in components.values() if str(c.get("type", "")).upper() == "A"]
    return max(ks) if ks else default


def _build_k_schedule(components, k_start=10.0, num_k_steps=10):
    k_final = _max_opamp_k(components)
    if k_final <= k_start:
        return [k_final]
    return list(np.logspace(np.log10(k_start), np.log10(k_final), num_k_steps))


def _voltage_indices(node_map):
    pairs = [(key, idx) for key, idx in node_map.items() if isinstance(key, int)]
    pairs.sort(key=lambda x: x[1])
    return np.array([idx for _, idx in pairs], dtype=int)


def _residual_norm(Y_csc, b, x, v_idx):
    r = Y_csc.dot(x) - b
    return float(np.max(np.abs(r[v_idx])))


def solve_nonlinear_circuit(
    Y_base,
    sources_base,
    components,
    node_map,
    V_ini,
    max_iter=100,
    tol=1e-6,
    num_steps=10,
    use_source_ramp=True,
    use_k_ramp=True,
    k_start=10.0,
    num_k_steps=10,
    k_schedule=None,
    # Backtracking candidates
    alphas=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02),
):
    """
    Newton with continuation + residual-based line search.

    ALSO IMPORTANT FOR SENSITIVITY:
      At the end, we re-stamp the Jacobian at the FINAL solution and build LU from it.
      This makes solve_adjoint() correct at the nonlinear operating point.
    """
    has_opamps = _has_opamps(components)
    v_idx = _voltage_indices(node_map)

    # Source ramp schedule
    if use_source_ramp:
        source_ramp = np.linspace(1.0 / num_steps, 1.0, num_steps)
    else:
        source_ramp = np.array([1.0])

    # k ramp schedule
    if has_opamps and use_k_ramp:
        if k_schedule is None:
            k_schedule = _build_k_schedule(components, k_start=k_start, num_k_steps=num_k_steps)
    else:
        k_schedule = [None]

    # Save original op-amp k values
    orig_k = {}
    if has_opamps:
        for name, comp in components.items():
            if str(comp.get("type", "")).upper() == "A":
                orig_k[name] = float(comp.get("k", 1e3))

    V_k = V_ini.copy()
    prev_V_k = V_ini.copy()
    lu = None

    # Track the final stage settings (for final Jacobian/LU rebuild)
    final_sources = sources_base.copy()
    final_k_eff = None

    for s_factor in source_ramp:
        if use_source_ramp:
            print(f"\n--- Ramping Source: {s_factor * 100:.1f}% ---")
        sources_s = sources_base.copy() * s_factor
        final_sources = sources_s.copy()  # will end at 100%

        for k_eff in k_schedule:
            final_k_eff = k_eff

            if has_opamps and use_k_ramp and (k_eff is not None):
                for name, comp in components.items():
                    if str(comp.get("type", "")).upper() == "A":
                        comp["k"] = float(k_eff)
                print(f"\n=== k-ramp stage: k = {k_eff:.3g} ===")

            for it in range(max_iter):
                # Stamp at current guess
                Y_iter = Y_base.tolil()
                b_iter = sources_s.copy()

                Y_iter, b_iter = stamp_nonlinear_components(
                    Y_iter, b_iter, components, node_map, prev_V_k, V_k
                )
                Y_iter = Y_iter.tocsc()

                # residual at current point
                r0 = _residual_norm(Y_iter, b_iter, V_k, v_idx)

                # Newton candidate (solve linearized system)
                lu_tmp, V_new = solve_linear_circuit(Y_iter, b_iter)

                # Backtracking line search
                accepted = False
                best = None

                for a in alphas:
                    V_trial = V_k + a * (V_new - V_k)

                    # Re-stamp to measure true nonlinear residual at trial
                    Y_t = Y_base.tolil()
                    b_t = sources_s.copy()
                    Y_t, b_t = stamp_nonlinear_components(
                        Y_t, b_t, components, node_map, V_k, V_trial
                    )
                    Y_t = Y_t.tocsc()

                    r_trial = _residual_norm(Y_t, b_t, V_trial, v_idx)

                    if best is None or r_trial < best[0]:
                        best = (r_trial, V_trial)

                    if r_trial <= 0.8 * r0 or r_trial < tol:
                        V_next = V_trial
                        accepted = True
                        break

                if not accepted:
                    V_next = best[1]

                max_error = float(np.max(np.abs(V_next[v_idx] - V_k[v_idx])))

                prev_V_k = V_k.copy()
                V_k = V_next

                print(f"Iteration: {it}, error = {max_error}")
                if max_error < tol:
                    print(f"Converged in {it + 1} iterations.")
                    break
            else:
                stage = ""
                if has_opamps and use_k_ramp and (k_eff is not None):
                    stage += f"k={k_eff:.3g}, "
                if use_source_ramp:
                    stage += f"src={s_factor * 100:.1f}%"
                raise RuntimeError(f"Newton-Raphson failed to converge at {stage}.")

    # === Rebuild FINAL Jacobian + LU at the final solution (critical for adjoint sensitivities) ===
    Y_final = Y_base.tolil()
    b_final = final_sources.copy()

    # Ensure op-amp k is at its final value before final Jacobian build
    if has_opamps and use_k_ramp and (final_k_eff is not None):
        for name, comp in components.items():
            if str(comp.get("type", "")).upper() == "A":
                comp["k"] = float(final_k_eff)

    # Stamp at (V_k, V_k) to build the Jacobian at the operating point
    Y_final, b_final = stamp_nonlinear_components(
        Y_final, b_final, components, node_map, V_k, V_k
    )
    Y_final = Y_final.tocsc()
    lu = solve_LU(Y_final)

    # Restore original op-amp k values
    if has_opamps:
        for name, k0 in orig_k.items():
            components[name]["k"] = k0

    return lu, V_k