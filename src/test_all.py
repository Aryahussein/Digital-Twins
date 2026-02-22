"""
Test script to verify all analysis modes work correctly.
Runs OP, AC, Transient, and Nonlinear OP tests with validation.
"""
import sys
import os
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from txt2dictionary import parse_netlist
from node_index import build_node_index
from simulations import run_op, run_ac_sweep, transient_analysis_loop
from assembleYmatrix import stamp_linear_components, stamp_static_components, initialize_stamps
from sources import evaluate_all_time_sources
from sensitivity import compute_step_sensitivities, aggregate_sweep_sensitivities, estimate_std_dev

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name} — {detail}")
        FAIL += 1

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol

# ============================================================
# TEST 1: Voltage Divider — OP Analysis
# ============================================================
print("\n" + "="*60)
print("TEST 1: Voltage Divider (OP)")
print("="*60)

components, analyses = parse_netlist("testfiles/voltage_divider.txt")
node_map = build_node_index(components)
comp_t0 = evaluate_all_time_sources(components, 0.0)

Y, sources = initialize_stamps(len(node_map), w=0)
Y, sources = stamp_linear_components(Y, sources, comp_t0, node_map)

VI, lu, sens = run_op(Y, sources, components, node_map, sensitivity=True, nonlinear=False, w=0)

check("Node 1 = 10V", approx(VI[node_map[1]], 10.0), f"got {VI[node_map[1]]:.6f}")
check("Node 2 = 5V", approx(VI[node_map[2]], 5.0), f"got {VI[node_map[2]]:.6f}")
check("I(V1) = -5mA", approx(VI[node_map["V1"]], -0.005), f"got {VI[node_map['V1']]:.6f}")
check("Sensitivity computed", sens is not None)
check("R1 sensitivity exists", "R1" in sens[1], f"keys: {list(sens[1].keys()) if 1 in sens else 'missing'}")
check("R2 sensitivity exists", "R2" in sens[1])
check("V1 sensitivity exists", "V1" in sens[1])

# Test estimate_std_dev with scalar sensitivities
if sens is not None and 2 in sens:
    try:
        std = estimate_std_dev(sens[2], components, percent_sigma=0.01)
        check("estimate_std_dev works on scalars", True)
    except Exception as e:
        check("estimate_std_dev works on scalars", False, str(e))

# ============================================================
# TEST 2: RC Low-Pass Filter — AC Analysis
# ============================================================
print("\n" + "="*60)
print("TEST 2: RC Low-Pass Filter (AC)")
print("="*60)

components, analyses = parse_netlist("testfiles/rc_lowpass.txt")
node_map = build_node_index(components)
comp_t0 = evaluate_all_time_sources(components, 0.0)

Y, sources = initialize_stamps(len(node_map), w=0)
Y, sources = stamp_linear_components(Y, sources, comp_t0, node_map)

freqs, VIs, lus, raw_sens = run_ac_sweep(
    Y, sources, components, node_map,
    start_freq=1, stop_freq=100000, points=50,
    output_nodes=[2], sensitivity=True
)

check("Frequency array correct length", len(freqs) == 50)
check("VI shape correct", VIs.shape == (50, len(node_map)), f"got {VIs.shape}")

# Check DC gain (at very low freq) — should be close to 1 (AC mag = 1V)
v_out_low = np.abs(VIs[0, node_map[2]])
check("DC gain ≈ 1.0", approx(v_out_low, 1.0, tol=0.05), f"got {v_out_low:.4f}")

# Check high-frequency rolloff — should be much less than 1
v_out_high = np.abs(VIs[-1, node_map[2]])
check("HF attenuation < 0.1", v_out_high < 0.1, f"got {v_out_high:.4f}")

# Find -3dB point (should be near 159 Hz)
mags = np.abs(VIs[:, node_map[2]])
mag_db = 20 * np.log10(np.where(mags == 0, 1e-12, mags))
dc_gain_db = mag_db[0]
idx_3db = np.argmin(np.abs(mag_db - (dc_gain_db - 3)))
f_3db = freqs[idx_3db]
check("f_3dB ≈ 159 Hz", approx(f_3db, 159.2, tol=30), f"got {f_3db:.1f} Hz")

# Sensitivity
if raw_sens is not None:
    sens_agg = aggregate_sweep_sensitivities(
        components, node_map, analyses, raw_sensitivities=raw_sens,
        output_nodes=[2]
    )
    check("AC sensitivity aggregated", "R1" in sens_agg[2])
    check("AC sensitivity array length", len(sens_agg[2]["R1"]) == 50, f"got {len(sens_agg[2]['R1'])}")

# ============================================================
# TEST 3: RC Transient — Step Response
# ============================================================
print("\n" + "="*60)
print("TEST 3: RC Transient (Step Response)")
print("="*60)

components, analyses = parse_netlist("testfiles/rc_transient.txt")
node_map = build_node_index(components)
comp_t0 = evaluate_all_time_sources(components, 0.0)

# CRITICAL FIX TEST: For transient, use stamp_static_components (not stamp_linear_components)
Y, sources = initialize_stamps(len(node_map), w=0)
Y, sources = stamp_static_components(Y, sources, comp_t0, node_map)

t_stop, dt = analyses[".TRAN"]["stop"], analyses[".TRAN"]["step"]
time, results, lus, raw_sens = transient_analysis_loop(
    Y, sources, components, node_map, t_stop, dt,
    output_nodes=[2], sensitivity=True
)

check("Time array length matches steps", len(time) == int(t_stop / dt))
check("Time[0] = 0", time[0] == 0.0)
check("Time matches step*dt", approx(time[5], 5 * dt, tol=1e-12), f"got {time[5]}")

# Check that capacitor voltage starts at 0
v_cap_start = results[0, node_map[2]]
check("V_cap(t=0) ≈ 0", approx(v_cap_start, 0.0, tol=0.1), f"got {v_cap_start:.4f}")

# After 5*tau = 5ms, should be near 5V (63.2% after 1*tau)
tau = 1e-3  # 1k * 1u
idx_5tau = int(5 * tau / dt)
if idx_5tau < len(results):
    v_cap_5tau = results[idx_5tau, node_map[2]]
    check("V_cap(5*tau) ≈ 5V (settled)", approx(v_cap_5tau, 5.0, tol=0.2), f"got {v_cap_5tau:.4f}")

# Check that source voltage is 5V during pulse
v_src_mid = results[int(2e-3 / dt), node_map[1]]  # At t=2ms, should be 5V
check("V_src(2ms) = 5V (during pulse)", approx(v_src_mid, 5.0, tol=0.01), f"got {v_src_mid:.4f}")

# Sensitivity
if raw_sens is not None:
    sens_agg = aggregate_sweep_sensitivities(
        components, node_map, analyses, raw_sensitivities=raw_sens,
        output_nodes=[2]
    )
    check("Transient sensitivity aggregated", "R1" in sens_agg[2])

# ============================================================
# TEST 4: Diode OP — Nonlinear
# ============================================================
print("\n" + "="*60)
print("TEST 4: Diode OP (Nonlinear)")
print("="*60)

components, analyses = parse_netlist("testfiles/diode_op.txt")
node_map = build_node_index(components)
comp_t0 = evaluate_all_time_sources(components, 0.0)

Y, sources = initialize_stamps(len(node_map), w=0)
Y, sources = stamp_linear_components(Y, sources, comp_t0, node_map)

try:
    VI, lu, sens = run_op(Y, sources, components, node_map, sensitivity=True, nonlinear=True, w=0)
    check("Nonlinear solve converged", True)
    
    # Diode should have ~0.6-0.7V across it
    v_diode = VI[node_map[2]]
    check("Diode voltage ≈ 0.6-0.7V", 0.5 < v_diode < 0.8, f"got {v_diode:.4f}")
    
    # Current through R1: (V1 - V2) / R1
    i_r1 = (VI[node_map[1]] - VI[node_map[2]]) / 1000
    check("Circuit current reasonable", 0.003 < i_r1 < 0.006, f"got {i_r1:.6f} A")
    
except RuntimeError as e:
    check("Nonlinear solve converged", False, str(e))

# ============================================================
# TEST 5: Parser edge cases
# ============================================================
print("\n" + "="*60)
print("TEST 5: Parser Tests")
print("="*60)

from txt2dictionary import parse_value, parse_tran_func

check("parse '1k' = 1000", parse_value("1k") == 1000.0)
check("parse '10MEG' = 1e7", parse_value("10MEG") == 1e7)
check("parse '100n' = 1e-7", approx(parse_value("100n"), 1e-7, tol=1e-20), f"got {repr(parse_value('100n'))}")
check("parse '2.2u' = 2.2e-6", approx(parse_value("2.2u"), 2.2e-6, tol=1e-12))
check("parse '1.5p' = 1.5e-12", approx(parse_value("1.5p"), 1.5e-12, tol=1e-18))

# Test SIN parsing with phase conversion
sin_result = parse_tran_func("SIN(0 1 1k 0 90)")
check("SIN parsed", sin_result is not None)
check("SIN phase converted to radians", approx(sin_result["PHASE"], np.pi/2, tol=1e-6),
      f"got {sin_result['PHASE']:.4f}")

# Test COS parsing
cos_result = parse_tran_func("COS(0 1 1k 0 0)")
check("COS parsed", cos_result is not None and cos_result["type"] == "COS")

# Test PULSE parsing  
pulse_result = parse_tran_func("PULSE(0 5 0 1n 1n 5u 10u)")
check("PULSE parsed", pulse_result is not None)
check("PULSE V2 = 5", pulse_result["V2"] == 5.0)

# ============================================================
# TEST 6: PULSE source edge cases
# ============================================================
print("\n" + "="*60)
print("TEST 6: PULSE Source Edge Cases")
print("="*60)

from sources import evaluate_time_source

# Test PULSE with TR=0, TF=0 (step function — should not crash)
step_pulse = {"type": "PULSE", "V1": 0, "V2": 5, "TD": 0, "TR": 0, "TF": 0, "PW": 5e-3, "PER": 10e-3}
try:
    v = evaluate_time_source(step_pulse, 1e-3)
    check("PULSE TR=0 TF=0 no crash", True)
    check("PULSE TR=0: high during PW", approx(v, 5.0, tol=0.01), f"got {v}")
except Exception as e:
    check("PULSE TR=0 TF=0 no crash", False, str(e))

# Test PULSE with PER=0 (single pulse, no repeat)
single_pulse = {"type": "PULSE", "V1": 0, "V2": 3, "TD": 1e-3, "TR": 0, "TF": 0, "PW": 2e-3, "PER": 0}
try:
    v_before = evaluate_time_source(single_pulse, 0.5e-3)  # Before delay
    v_during = evaluate_time_source(single_pulse, 2e-3)     # During pulse
    check("PULSE PER=0 no crash", True)
    check("PULSE PER=0: V1 before delay", approx(v_before, 0.0), f"got {v_before}")
except Exception as e:
    check("PULSE PER=0 no crash", False, str(e))

# ============================================================
# TEST 7: Transient sensitivity is non-zero for C
# ============================================================
print("\n" + "="*60)
print("TEST 7: Transient C/L Sensitivity (non-zero)")
print("="*60)

# Re-use results from Test 3 (RC transient)
if raw_sens is not None and len(raw_sens) > 10:
    # Check that C1 sensitivity is not all zeros (this was the bug)
    c1_sens_sample = raw_sens[100][2]["C1"] if 2 in raw_sens[100] else None
    if c1_sens_sample is not None:
        check("C1 transient sensitivity non-zero", abs(c1_sens_sample) > 1e-20, f"got {c1_sens_sample}")
    else:
        check("C1 transient sensitivity non-zero", False, "sensitivity data missing")
    
    r1_sens_sample = raw_sens[100][2]["R1"] if 2 in raw_sens[100] else None
    if r1_sens_sample is not None:
        check("R1 transient sensitivity non-zero", abs(r1_sens_sample) > 1e-20, f"got {r1_sens_sample}")
else:
    check("C1 transient sensitivity non-zero", False, "no raw_sens data from test 3")

# ============================================================
# TEST 8: Node validation
# ============================================================
print("\n" + "="*60)
print("TEST 8: Node Validation")
print("="*60)

from node_index import validate_node

# Test existing nodes
try:
    idx = validate_node(1, node_map)
    check("validate_node(1) succeeds", idx is not None)
except KeyError:
    check("validate_node(1) succeeds", False)

# Test ground
idx = validate_node(0, node_map)
check("validate_node(0) returns None (ground)", idx is None)

# Test missing node
try:
    validate_node(999, node_map)
    check("validate_node(999) raises KeyError", False, "no exception raised")
except KeyError:
    check("validate_node(999) raises KeyError", True)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
print("="*60)

if FAIL > 0:
    sys.exit(1)
