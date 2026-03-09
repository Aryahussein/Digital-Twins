"""
Adjoint Applications (Lecture 9 — Ron Rohrer, UTD Spring 2026)

Implements the six adjoint-based analysis techniques described in the lecture:

  1. Noise Estimation       — device noise → output noise via adjoint transfer
  2. Yield Estimation       — σ_output from parameter tolerances + sensitivities
  3. Fault Dictionary       — rank parameters by impact on output
  4. Distortion Analysis    — harmonic distortion via adjoint at each harmonic
  5. Design Centering       — gradient-based shift toward specification center
  6. Radiation (stub)       — placeholder for nonlinear time-domain adjoint

All routines take the same core inputs that already exist in the simulator:
    - components, node_map, VI (forward solution), lu (LU factorization)
    - and use solver.solve_adjoint + sensitivity.get_all_sensitivities
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

from solver import solve_adjoint
from sensitivity import (
    get_all_sensitivities,
    compute_step_sensitivities,
    estimate_std_dev,
)
from constants import kb, T, e, Vt

logger = logging.getLogger(__name__)


# =============================================================================
# 1. NOISE ESTIMATION
# =============================================================================
#
# Lecture slide 6, item 1:
#   "A single analysis of the adjoint circuit determines the transfer from
#    all devices back to the output."
#
# Each device contributes noise (thermal, shot, flicker). The adjoint vector
# Ψ gives how much each device's noise current/voltage transfers to the
# output. Total output noise = Σ |Ψ_k|² · S_noise_k(f)
#


def _resistor_thermal_noise_psd(R, temp=T):
    """Thermal noise PSD of a resistor: S_i = 4kT/R [A²/Hz]."""
    return 4.0 * kb * temp / R


def _diode_shot_noise_psd(Id):
    """Shot noise PSD of a diode: S_i = 2qI [A²/Hz]."""
    return 2.0 * e * abs(Id)


def _mosfet_thermal_noise_psd(gm, gamma=2.0 / 3.0, temp=T):
    """
    MOSFET channel thermal noise: S_id = 4kT·γ·gm [A²/Hz].
    γ = 2/3 for long-channel, can be higher for short-channel.
    """
    return 4.0 * kb * temp * gamma * gm


def _mosfet_gm_from_op(VI, comp, node_map):
    """Extract gm from the DC operating point for a MOSFET."""
    params = comp.get("model_params", {})
    inst = comp.get("inst_params", {})
    m_type = comp.get("model_type", "NMOS")

    VTO = float(
        params.get("VTO", params.get("VT0", params.get("VTH", params.get("VTH0", 0.7))))
    )
    W = float(inst.get("W", 1.0))
    L = max(float(inst.get("L", 1.0)), 1e-12)

    KP = float(params.get("KP", 0.0))
    if KP == 0.0:
        mu = float(params.get("MU", params.get("UO", params.get("U0", 0.0))))
        cox = float(params.get("C_OX", params.get("COX", 0.0)))
        KP = mu * cox

    Bn = (W / L) * KP

    idx_g = node_map.get(comp["n_g"])
    idx_s = node_map.get(comp["n_s"])
    idx_d = node_map.get(comp["n_d"])

    vg = VI[idx_g] if idx_g is not None else 0.0
    vs = VI[idx_s] if idx_s is not None else 0.0
    vd = VI[idx_d] if idx_d is not None else 0.0

    if m_type == "PMOS":
        vov = (vs - vg) - abs(VTO)
    else:
        vov = (vg - vs) - VTO

    if vov <= 0.0:
        return 0.0

    if m_type == "PMOS":
        vds_eff = vs - vd
    else:
        vds_eff = vd - vs

    if vds_eff < vov:
        return Bn * vds_eff   # triode
    else:
        return Bn * vov       # saturation


def compute_output_noise(
    components: dict,
    node_map: dict,
    VI: np.ndarray,
    lu,
    output_node,
    freq: float = 1000.0,
    temp: float = T,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total output-referred noise spectral density at a given frequency.

    Uses the adjoint vector to transfer each device's noise contribution
    to the output node, exactly as described in the lecture:

        V²_noise_out(f) = Σ_k  |Ψ_k|² · S_noise_k(f)

    where Ψ is the adjoint solution with excitation at the output node.

    Args:
        components: parsed component dict
        node_map: node-to-index mapping
        VI: DC operating point solution vector
        lu: LU factorization from OP solve
        output_node: which node to compute output noise at
        freq: frequency in Hz for noise PSD
        temp: temperature in Kelvin

    Returns:
        total_noise_density: total output noise PSD [V²/Hz]
        contributions: dict {component_name: noise_psd_contribution}
    """
    # Solve adjoint: Ψ = Y^(-T) · e_output
    Psi = solve_adjoint(lu, output_node, node_map)

    contributions = {}

    for name, comp in components.items():
        ctype = comp["type"]
        n1 = comp.get("n1", 0)
        n2 = comp.get("n2", 0)
        idx1 = node_map.get(n1)
        idx2 = node_map.get(n2)

        # Adjoint transfer: voltage difference seen by this device
        psi1 = Psi[idx1] if idx1 is not None else 0.0
        psi2 = Psi[idx2] if idx2 is not None else 0.0
        psi_diff = psi1 - psi2

        if ctype == "R":
            R = comp["value"]
            S_noise = _resistor_thermal_noise_psd(R, temp)
            # Noise current source in parallel: output contribution = |Ψ|² · S_i
            contributions[name] = float(abs(psi_diff) ** 2 * S_noise)

        elif ctype == "D":
            # Diode shot noise based on DC bias current
            v1 = VI[idx1] if idx1 is not None else 0.0
            v2 = VI[idx2] if idx2 is not None else 0.0
            vd = v1 - v2

            params = comp.get("model_params", {})
            Is = float(params.get("IS", comp.get("value", 1e-14)))
            N = float(params.get("N", 1.0))
            Id = Is * (np.exp(np.clip(vd / (N * Vt), -50, 50)) - 1.0)

            S_noise = _diode_shot_noise_psd(Id)
            contributions[name] = float(abs(psi_diff) ** 2 * S_noise)

        elif ctype == "M":
            nd, ns = comp["n_d"], comp["n_s"]
            idx_d = node_map.get(nd)
            idx_s = node_map.get(ns)

            psi_d = Psi[idx_d] if idx_d is not None else 0.0
            psi_s = Psi[idx_s] if idx_s is not None else 0.0
            psi_ds = psi_d - psi_s

            gm = _mosfet_gm_from_op(VI, comp, node_map)
            S_noise = _mosfet_thermal_noise_psd(gm, temp=temp)
            contributions[name] = float(abs(psi_ds) ** 2 * S_noise)

        elif ctype == "Q":
            # BJT shot noise: 2qIc on collector and 2qIb on base
            nc, nb, ne = comp.get("n_c", 0), comp.get("n_b", 0), comp.get("n_e", 0)
            idx_c = node_map.get(nc)
            idx_b = node_map.get(nb)
            idx_e = node_map.get(ne)

            psi_c = Psi[idx_c] if idx_c is not None else 0.0
            psi_b = Psi[idx_b] if idx_b is not None else 0.0
            psi_e = Psi[idx_e] if idx_e is not None else 0.0
            psi_ce = psi_c - psi_e
            psi_be = psi_b - psi_e

            params = comp.get("model_params", {})
            IS = float(params.get("IS", 1e-14))
            BF = float(params.get("BF", 100.0))
            NF = float(params.get("NF", 1.0))

            vc = VI[idx_c] if idx_c is not None else 0.0
            vb = VI[idx_b] if idx_b is not None else 0.0
            ve = VI[idx_e] if idx_e is not None else 0.0

            m_type = comp.get("model_type", "NPN")
            if m_type == "PNP":
                vbe = ve - vb
            else:
                vbe = vb - ve

            Vte = max(NF * Vt, 1e-12)
            Ic = IS * (np.exp(np.clip(vbe / Vte, -50, 50)) - 1.0)
            Ib = Ic / max(BF, 1.0)

            S_ic = _diode_shot_noise_psd(Ic)  # 2qIc
            S_ib = _diode_shot_noise_psd(Ib)  # 2qIb

            noise_c = abs(psi_ce) ** 2 * S_ic
            noise_b = abs(psi_be) ** 2 * S_ib
            contributions[name] = float(noise_c + noise_b)

        # V, I, C, L, G, O, E, F, H: assumed noiseless

    total = sum(contributions.values())
    return total, contributions


# =============================================================================
# 2. YIELD ESTIMATION
# =============================================================================
#
# Lecture slide 6, item 2:
#   Use sensitivity to estimate how parameter variations propagate to the
#   output. If each parameter p_k has tolerance σ_k, then:
#
#       σ²_output = Σ_k (∂output/∂p_k · σ_k)²
#
#   Yield ≈ probability that output stays within spec limits.
#


def estimate_yield(
    sensitivities: Dict[str, complex],
    components: dict,
    spec_min: float,
    spec_max: float,
    nominal_output: float,
    tolerances: Optional[Dict[str, float]] = None,
    default_tolerance_pct: float = 0.01,
) -> Dict[str, object]:
    """
    Estimate manufacturing yield using adjoint sensitivities.

    Computes the output standard deviation from parameter tolerances,
    then estimates the probability that the output stays within specs
    assuming Gaussian parameter distributions.

    Args:
        sensitivities: {param_key: dOutput/dParam} from get_all_sensitivities
        components: parsed component dict
        spec_min: lower specification limit for the output
        spec_max: upper specification limit for the output
        nominal_output: the nominal (simulated) output value
        tolerances: optional {param_key: sigma_value} overrides
        default_tolerance_pct: default percentage tolerance (0.01 = 1%)

    Returns:
        dict with keys: sigma_output, yield_estimate, margin_low, margin_high,
                        cpk, top_contributors
    """
    if tolerances is None:
        tolerances = {}

    variance = 0.0
    contrib_list = []

    for key, sens in sensitivities.items():
        if isinstance(sens, np.ndarray) and sens.size == 0:
            continue
        s = float(np.abs(sens))
        if s == 0:
            continue

        # Get parameter sigma
        if key in tolerances:
            sigma_p = tolerances[key]
        else:
            # Derive from nominal value × percentage
            if ":" in key:
                dev, pname = key.split(":", 1)
                comp = components.get(dev, {})
                params = comp.get("model_params", {})
                base = float(params.get(pname, params.get(pname.upper(), comp.get("value", 0.0))))
            else:
                base = float(components.get(key, {}).get("value", 0.0))
            sigma_p = abs(base) * default_tolerance_pct

        contribution = (s * sigma_p) ** 2
        variance += contribution
        contrib_list.append((key, contribution))

    sigma_output = np.sqrt(variance) if variance > 0 else 1e-30

    # Margins in units of sigma
    margin_low = (nominal_output - spec_min) / sigma_output
    margin_high = (spec_max - nominal_output) / sigma_output

    # Cpk (process capability index)
    cpk = min(margin_low, margin_high) / 3.0

    # Yield estimate using normal CDF
    from scipy.stats import norm

    yield_low = norm.cdf(margin_low)
    yield_high = norm.cdf(margin_high)
    yield_est = max(0.0, yield_low + yield_high - 1.0)

    # Top contributors (sorted by variance contribution)
    contrib_list.sort(key=lambda x: x[1], reverse=True)
    top = [(k, np.sqrt(v) / sigma_output * 100.0) for k, v in contrib_list[:10]]

    return {
        "sigma_output": sigma_output,
        "yield_estimate": yield_est,
        "yield_ppm": (1.0 - yield_est) * 1e6,
        "margin_low_sigma": margin_low,
        "margin_high_sigma": margin_high,
        "cpk": cpk,
        "top_contributors": top,  # list of (param_name, % of total sigma)
    }


# =============================================================================
# 3. FAULT DICTIONARY FOR TEST
# =============================================================================
#
# Lecture slide 6, item 3:
#   The sensitivity tells us how much each parameter shift changes the output.
#   A "fault dictionary" catalogs, for each component, what fault signature
#   (output shift) it would produce — enabling diagnosis from measured output.
#


def build_fault_dictionary(
    sensitivities: Dict[str, complex],
    components: dict,
    fault_magnitudes: Optional[Dict[str, List[float]]] = None,
    default_faults_pct: Optional[List[float]] = None,
) -> Dict[str, List[Dict]]:
    """
    Build a fault dictionary using adjoint sensitivities.

    For each component, predict the output shift under various fault
    conditions (e.g., ±10%, ±50%, open, short) using first-order
    sensitivity approximation:

        ΔOutput ≈ (∂Output/∂p) · Δp

    This creates a lookup table: given a measured output deviation,
    which component fault best explains it?

    Args:
        sensitivities: {param_key: dOutput/dParam}
        components: parsed component dict
        fault_magnitudes: optional per-component fault Δp values
        default_faults_pct: default fault percentages to test
            (e.g., [-0.5, -0.1, +0.1, +0.5] for ±10%, ±50%)

    Returns:
        fault_dict: {component_name: [{"fault": description,
                                        "delta_p": float,
                                        "delta_output": float}, ...]}
    """
    if default_faults_pct is None:
        default_faults_pct = [-0.50, -0.10, +0.10, +0.50]

    fault_dict = {}

    for key, sens in sensitivities.items():
        if isinstance(sens, np.ndarray):
            if sens.size == 0:
                continue
            sens = sens[0]  # use first point for dictionary

        s = complex(sens)

        # Get nominal value
        if ":" in key:
            dev, pname = key.split(":", 1)
            comp = components.get(dev, {})
            params = comp.get("model_params", {})
            nominal = float(
                params.get(pname, params.get(pname.upper(), comp.get("value", 0.0)))
            )
        else:
            nominal = float(components.get(key, {}).get("value", 0.0))

        if nominal == 0.0:
            continue

        # Get fault magnitudes for this component
        if fault_magnitudes and key in fault_magnitudes:
            deltas = fault_magnitudes[key]
        else:
            deltas = [pct * nominal for pct in default_faults_pct]

        entries = []
        for dp in deltas:
            pct = dp / nominal * 100 if nominal != 0 else 0
            delta_out = s * dp
            entries.append(
                {
                    "fault": f"{pct:+.0f}% ({nominal:.4g} → {nominal + dp:.4g})",
                    "delta_p": dp,
                    "delta_output": float(np.real(delta_out)),
                    "delta_output_mag": float(np.abs(delta_out)),
                }
            )

        fault_dict[key] = entries

    return fault_dict


def diagnose_fault(
    fault_dict: Dict[str, List[Dict]],
    measured_delta_output: float,
    tolerance: float = 0.2,
) -> List[Tuple[str, Dict]]:
    """
    Given a measured output deviation, find the most likely faulty component.

    Searches the fault dictionary for entries whose predicted delta_output
    is closest to the measured deviation.

    Args:
        fault_dict: output of build_fault_dictionary
        measured_delta_output: observed output shift from nominal
        tolerance: relative matching tolerance (0.2 = 20%)

    Returns:
        List of (component_key, fault_entry) sorted by closeness of match.
    """
    candidates = []
    for key, entries in fault_dict.items():
        for entry in entries:
            predicted = entry["delta_output"]
            if predicted == 0:
                continue
            error = abs(predicted - measured_delta_output) / abs(measured_delta_output + 1e-30)
            if error < tolerance:
                candidates.append((key, entry, error))

    candidates.sort(key=lambda x: x[2])
    return [(k, e) for k, e, _ in candidates]


# =============================================================================
# 4. DISTORTION ANALYSIS
# =============================================================================
#
# Lecture slide 6, item 4:
#   "Forward solve → Use amplitude at each device to determine harmonic
#    distortions → Solve adjoint circuit at each harmonic to determine the
#    contribution of each device to output distortion."
#
# Approach: small-signal AC solve at fundamental frequency f0, then use the
# signal amplitude at each nonlinear device to estimate its harmonic
# generation (2nd and 3rd order), then solve the adjoint at 2f0 and 3f0
# to find how those harmonics transfer to the output.
#


def _diode_harmonic_currents(vd_amplitude, Is=1e-14, N=1.0, Vdc=0.6):
    """
    Estimate 2nd and 3rd harmonic current amplitudes for a diode
    using Taylor expansion of I = Is·exp(V/NVt).

    At bias point Vdc with small-signal amplitude va:
        I(Vdc + va·cos(wt)) ≈ Idc + i1·cos(wt) + i2·cos(2wt) + i3·cos(3wt)

    where the harmonic amplitudes come from the exponential Taylor series.
    """
    Vte = N * Vt
    exp_dc = np.exp(np.clip(Vdc / Vte, -50, 50))
    gm = Is * exp_dc / Vte

    va = abs(vd_amplitude)
    x = va / Vte

    # From Taylor expansion of exp(x·cos(θ)) using modified Bessel functions
    # Simplified for small x: I_n ≈ gm · (x/2)^n / n!
    i2 = gm * (x / 2.0) ** 2 / 2.0   # 2nd harmonic
    i3 = gm * (x / 2.0) ** 3 / 6.0   # 3rd harmonic

    return i2, i3


def estimate_harmonic_distortion(
    components: dict,
    node_map: dict,
    VI_dc: np.ndarray,
    VI_ac: np.ndarray,
    lu_2f0,
    lu_3f0,
    output_node,
    freq: float,
) -> Dict[str, object]:
    """
    Estimate harmonic distortion using the adjoint method.

    Procedure (from lecture):
    1. Forward AC solve at f0 gives signal amplitude at each device
    2. Nonlinear device models give harmonic current amplitudes
    3. Adjoint at 2f0 and 3f0 gives transfer to output
    4. HD2, HD3, THD computed from the contributions

    Args:
        components: parsed component dict
        node_map: node-to-index map
        VI_dc: DC operating point
        VI_ac: AC solution at fundamental frequency f0
        lu_2f0: LU factorization at 2·f0
        lu_3f0: LU factorization at 3·f0
        output_node: target output node
        freq: fundamental frequency f0

    Returns:
        dict with HD2, HD3, THD, and per-device contributions
    """
    # Adjoint vectors at harmonic frequencies
    Psi_2f = solve_adjoint(lu_2f0, output_node, node_map)
    Psi_3f = solve_adjoint(lu_3f0, output_node, node_map)

    # Output fundamental amplitude
    out_idx = node_map[output_node]
    V_fund = abs(VI_ac[out_idx])
    if V_fund == 0:
        return {"HD2": 0.0, "HD3": 0.0, "THD": 0.0, "device_contributions": {}}

    V_2f_total = 0.0 + 0j
    V_3f_total = 0.0 + 0j
    device_contribs = {}

    for name, comp in components.items():
        ctype = comp["type"]

        if ctype == "D":
            n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
            idx1 = node_map.get(n1)
            idx2 = node_map.get(n2)

            # AC signal amplitude across diode
            va1 = VI_ac[idx1] if idx1 is not None else 0.0
            va2 = VI_ac[idx2] if idx2 is not None else 0.0
            va = va1 - va2

            # DC bias
            vdc1 = VI_dc[idx1] if idx1 is not None else 0.0
            vdc2 = VI_dc[idx2] if idx2 is not None else 0.0
            vdc = vdc1 - vdc2

            params = comp.get("model_params", {})
            Is = float(params.get("IS", comp.get("value", 1e-14)))
            N_coeff = float(params.get("N", 1.0))

            i2, i3 = _diode_harmonic_currents(va, Is, N_coeff, vdc)

            # Adjoint transfer at each harmonic
            psi1_2f = Psi_2f[idx1] if idx1 is not None else 0.0
            psi2_2f = Psi_2f[idx2] if idx2 is not None else 0.0
            psi1_3f = Psi_3f[idx1] if idx1 is not None else 0.0
            psi2_3f = Psi_3f[idx2] if idx2 is not None else 0.0

            V_2f_dev = i2 * (psi1_2f - psi2_2f)
            V_3f_dev = i3 * (psi1_3f - psi2_3f)

            V_2f_total += V_2f_dev
            V_3f_total += V_3f_dev
            device_contribs[name] = {
                "HD2_contribution": float(abs(V_2f_dev) / V_fund),
                "HD3_contribution": float(abs(V_3f_dev) / V_fund),
            }

        # MOSFET distortion could be added similarly using square-law
        # Taylor expansion, but that requires more detailed modeling

    HD2 = float(abs(V_2f_total) / V_fund)
    HD3 = float(abs(V_3f_total) / V_fund)
    THD = float(np.sqrt(HD2**2 + HD3**2))

    return {
        "fundamental_amplitude": float(V_fund),
        "HD2": HD2,
        "HD3": HD3,
        "THD": THD,
        "HD2_dB": 20 * np.log10(max(HD2, 1e-30)),
        "HD3_dB": 20 * np.log10(max(HD3, 1e-30)),
        "THD_dB": 20 * np.log10(max(THD, 1e-30)),
        "device_contributions": device_contribs,
    }


# =============================================================================
# 5. DESIGN CENTERING
# =============================================================================
#
# Lecture slide 7:
#   "Each I_k takes performance to the edge of a specification.
#    Gradients with respect to all specification planes meet at design center."
#
# The idea: use sensitivity gradients to shift the design point so that it
# is maximally distant from all specification boundaries (maximize yield).
#


def compute_design_center_step(
    sensitivities: Dict[str, complex],
    components: dict,
    nominal_output: float,
    spec_min: float,
    spec_max: float,
    learning_rate: float = 0.1,
    tolerances: Optional[Dict[str, float]] = None,
    default_tolerance_pct: float = 0.01,
) -> Dict[str, float]:
    """
    Compute one design centering step using sensitivity gradients.

    The goal is to adjust parameters so the nominal output moves toward
    the center of the specification window [spec_min, spec_max], weighted
    by how much each parameter contributes to output variance.

    This implements the gradient-based approach from the lecture: the
    sensitivity vector defines the "direction" in parameter space that
    moves the output, and we step along it to center the output.

    Args:
        sensitivities: {param_key: dOutput/dParam}
        components: parsed component dict
        nominal_output: current output value
        spec_min, spec_max: specification limits
        learning_rate: step size scaling factor
        tolerances: parameter sigma values (for weighting)
        default_tolerance_pct: default tolerance percentage

    Returns:
        param_deltas: {param_key: recommended_delta_p} for each tunable parameter
    """
    spec_center = (spec_min + spec_max) / 2.0
    output_error = spec_center - nominal_output  # how far we are from center

    if abs(output_error) < 1e-15:
        return {}

    # Build gradient vector: which parameters to adjust and by how much
    param_deltas = {}
    total_gradient_sq = 0.0

    for key, sens in sensitivities.items():
        if isinstance(sens, np.ndarray):
            continue  # skip sweep data
        s = float(np.real(sens))
        if abs(s) < 1e-30:
            continue

        # Weight by inverse tolerance (parameters with tight tolerance
        # should move less)
        if ":" in key:
            dev, pname = key.split(":", 1)
            comp = components.get(dev, {})
            params = comp.get("model_params", {})
            nominal_p = float(params.get(pname, comp.get("value", 0.0)))
        else:
            nominal_p = float(components.get(key, {}).get("value", 0.0))

        if abs(nominal_p) < 1e-30:
            continue

        sigma_p = abs(nominal_p) * default_tolerance_pct
        if tolerances and key in tolerances:
            sigma_p = tolerances[key]

        # Gradient in normalized parameter space
        grad = s * sigma_p
        total_gradient_sq += grad**2

    if total_gradient_sq < 1e-30:
        return {}

    # Step along the gradient direction
    for key, sens in sensitivities.items():
        if isinstance(sens, np.ndarray):
            continue
        s = float(np.real(sens))
        if abs(s) < 1e-30:
            continue

        if ":" in key:
            dev, pname = key.split(":", 1)
            comp = components.get(dev, {})
            params = comp.get("model_params", {})
            nominal_p = float(params.get(pname, comp.get("value", 0.0)))
        else:
            nominal_p = float(components.get(key, {}).get("value", 0.0))

        if abs(nominal_p) < 1e-30:
            continue

        sigma_p = abs(nominal_p) * default_tolerance_pct
        if tolerances and key in tolerances:
            sigma_p = tolerances[key]

        grad = s * sigma_p
        # Steepest descent step in normalized space, then un-normalize
        delta_p = learning_rate * output_error * grad / total_gradient_sq * sigma_p
        param_deltas[key] = delta_p

    return param_deltas


# =============================================================================
# 6. RADIATION (Stub)
# =============================================================================
#
# Lecture slide 8:
#   "Difficult but Doable — Nonlinear and Time Domain"
#
# Full implementation requires time-domain adjoint (solving the adjoint
# system backwards in time), which is significantly more complex.
# This stub provides the interface for future implementation.
#

def estimate_radiation_sensitivity(
    components: dict,
    node_map: dict,
    time_array: np.ndarray,
    VI_transient: np.ndarray,
    list_of_lus: list,
    output_node,
    radiation_params: Optional[Dict] = None,
):
    """
    Placeholder for radiation sensitivity analysis.

    Full implementation would:
    1. Run forward transient simulation (done by caller)
    2. Inject radiation-induced photocurrents at each device
    3. Solve the time-reversed adjoint transient to find how each
       device's radiation response transfers to the output
    4. Integrate over time for total dose or single-event effects

    This requires backwards-in-time adjoint solving which is not yet
    implemented in the transient solver.

    Raises:
        NotImplementedError with description of what's needed.
    """
    raise NotImplementedError(
        "Radiation sensitivity requires time-domain adjoint (backwards-in-time "
        "integration of the adjoint system). This is planned for a future release. "
        "The forward transient infrastructure and per-step LU factors are available; "
        "the missing piece is the reverse-time adjoint sweep with proper terminal "
        "conditions."
    )


# =============================================================================
# CONVENIENCE: Run all applicable analyses at once
# =============================================================================


def run_adjoint_applications(
    components: dict,
    node_map: dict,
    VI: np.ndarray,
    lu,
    output_node,
    sensitivities: Optional[Dict[str, complex]] = None,
    spec_min: Optional[float] = None,
    spec_max: Optional[float] = None,
    freq: float = 1000.0,
    temp: float = T,
) -> Dict[str, object]:
    """
    Run all applicable adjoint analyses in one call.

    Returns a dict with results for noise, yield (if specs given),
    fault dictionary, and design centering (if specs given).
    """
    results = {}

    # Always compute sensitivities if not provided
    if sensitivities is None:
        from sensitivity import compute_step_sensitivities

        step_sens, _ = compute_step_sensitivities(
            lu, VI, components, node_map, [output_node]
        )
        sensitivities = step_sens.get(output_node, {})

    # 1. Noise
    total_noise, noise_contribs = compute_output_noise(
        components, node_map, VI, lu, output_node, freq=freq, temp=temp
    )
    results["noise"] = {
        "total_V2_per_Hz": total_noise,
        "total_Vrms_per_rtHz": np.sqrt(max(total_noise, 0)),
        "contributions": noise_contribs,
    }
    logger.info(
        "Output noise at %.0f Hz: %.3e V²/Hz (%.3e V/√Hz)",
        freq,
        total_noise,
        np.sqrt(max(total_noise, 0)),
    )

    # 2. Yield (if specs provided)
    if spec_min is not None and spec_max is not None:
        out_idx = node_map[output_node]
        nominal = float(np.real(VI[out_idx]))
        yield_result = estimate_yield(
            sensitivities, components, spec_min, spec_max, nominal
        )
        results["yield"] = yield_result
        logger.info(
            "Yield estimate: %.4f%% (Cpk=%.2f, σ_out=%.3e)",
            yield_result["yield_estimate"] * 100,
            yield_result["cpk"],
            yield_result["sigma_output"],
        )

    # 3. Fault dictionary
    fault_dict = build_fault_dictionary(sensitivities, components)
    results["fault_dictionary"] = fault_dict

    # 5. Design centering (if specs provided)
    if spec_min is not None and spec_max is not None:
        out_idx = node_map[output_node]
        nominal = float(np.real(VI[out_idx]))
        deltas = compute_design_center_step(
            sensitivities, components, nominal, spec_min, spec_max
        )
        results["design_centering"] = deltas

    return results
