"""
Comprehensive Test Suite for the SPICE Simulator.

Tests all analysis types, component types, parser fixes, and edge cases
identified in the code review.
"""

import sys
import os
import traceback
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.parser import NetlistParser
from core.circuit import Circuit
from engines.simulator import Simulator
from core.results import SimulationResult
from main import run_simulation_core

PASS = 0
FAIL = 0

def test(name, func):
    global PASS, FAIL
    try:
        func()
        PASS += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        print()

# ======================================================================
# 1. PARSER TESTS
# ======================================================================
print("\n" + "="*60)
print("1. PARSER TESTS")
print("="*60)

def test_parse_value_valid():
    p = NetlistParser()
    # Use np.isclose for floating-point multiplication results
    assert np.isclose(p._parse_value("10k"), 10e3)
    assert np.isclose(p._parse_value("5u"), 5e-6)
    assert np.isclose(p._parse_value("1.5MEG"), 1.5e6)
    assert np.isclose(p._parse_value("100"), 100.0)
    assert np.isclose(p._parse_value("3.3"), 3.3)
    assert np.isclose(p._parse_value("1e-3"), 1e-3)
    assert np.isclose(p._parse_value("47p"), 47e-12)

test("_parse_value with valid SPICE suffixes", test_parse_value_valid)

def test_parse_value_invalid():
    """Fix 4.1: _parse_value should raise ValueError on garbage input."""
    p = NetlistParser()
    try:
        p._parse_value("abc")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

test("_parse_value raises on invalid input (Fix 4.1)", test_parse_value_invalid)

def test_dc_sweep_parsing():
    """Fix 1.1: .DC command must produce a '.DC' key, not '.OP'."""
    p = NetlistParser()
    # Write a temp netlist
    path = "testfiles/_test_dc.txt"
    with open(path, "w") as f:
        f.write("V1 1 0 DC 5\nR1 1 2 1k\nR2 2 0 1k\n.DC V1 0 10 0.5\n")
    comps, analyses = p.parse(path)
    assert ".DC" in analyses, f"Expected '.DC' key, got: {list(analyses.keys())}"
    assert analyses[".DC"]["source"] == "V1"
    assert analyses[".DC"]["start"] == 0.0
    assert analyses[".DC"]["stop"] == 10.0
    assert analyses[".DC"]["step"] == 0.5
    assert ".OP" not in analyses, ".DC should NOT create a .OP key"
    os.remove(path)

test(".DC sweep parsing produces correct keys (Fix 1.1)", test_dc_sweep_parsing)

def test_op_parsing():
    """Ensure .OP still works correctly after .DC fix."""
    p = NetlistParser()
    path = "testfiles/_test_op.txt"
    with open(path, "w") as f:
        f.write("V1 1 0 DC 5\nR1 1 0 1k\n.OP\n")
    comps, analyses = p.parse(path)
    assert ".OP" in analyses
    assert ".DC" not in analyses
    os.remove(path)

test(".OP parsing still works after .DC fix", test_op_parsing)

def test_tran_tstart_parsed():
    """Fix 2.1: TSTART should be parsed and stored."""
    p = NetlistParser()
    path = "testfiles/_test_tran.txt"
    with open(path, "w") as f:
        f.write("V1 1 0 DC 5\nR1 1 0 1k\n.TRAN 1u 10m 5m\n")
    comps, analyses = p.parse(path)
    assert ".TRAN" in analyses
    assert "start" in analyses[".TRAN"]
    assert analyses[".TRAN"]["start"] == 5e-3
    os.remove(path)

test(".TRAN TSTART parameter parsed (Fix 2.1)", test_tran_tstart_parsed)

def test_vcvs_parser():
    """Fix 1.4: E-prefix components must be parsed (VCVS/OpAmp)."""
    p = NetlistParser()
    path = "testfiles/_test_opamp.txt"
    with open(path, "w") as f:
        f.write("V1 1 0 DC 1\nR1 1 2 1k\nR2 2 0 1k\nE1 3 0 2 0 100000\nR3 3 0 1MEG\n.OP\n")
    comps, analyses = p.parse(path)
    assert "E1" in comps, f"E1 not found in components: {list(comps.keys())}"
    assert comps["E1"]["type"] == "E"
    assert comps["E1"]["n_out"] == 3
    assert comps["E1"]["value"] == 100000.0
    os.remove(path)

test("E-prefix (VCVS/OpAmp) parser exists (Fix 1.4)", test_vcvs_parser)

def test_parser_reset():
    """Fix 6.1: Repeated parse calls should not accumulate state."""
    p = NetlistParser()
    path1 = "testfiles/_test_r1.txt"
    path2 = "testfiles/_test_r2.txt"
    with open(path1, "w") as f:
        f.write("V1 1 0 DC 5\nR1 1 0 1k\n.OP\n")
    with open(path2, "w") as f:
        f.write("V2 1 0 DC 3\nR2 1 0 2k\n.OP\n")
    c1, a1 = p.parse(path1)
    c2, a2 = p.parse(path2)
    assert "R1" not in c2, "State from first parse leaked into second"
    assert "V1" not in c2
    assert "R2" in c2
    os.remove(path1)
    os.remove(path2)

test("Parser state resets between parse() calls (Fix 6.1)", test_parser_reset)

# ======================================================================
# 2. COMPONENT / CIRCUIT TESTS
# ======================================================================
print("\n" + "="*60)
print("2. COMPONENT & CIRCUIT TESTS")
print("="*60)

def test_resistor_zero_value():
    """Fix 4.2: Zero-ohm resistor should raise ValueError."""
    try:
        from components.resistor import Resistor
        r = Resistor("R_BAD", {"type": "R", "n1": 1, "n2": 0, "value": 0.0})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "0 Ohms" in str(e)

test("Zero-ohm resistor raises ValueError (Fix 4.2)", test_resistor_zero_value)

def test_inductor_zero_value():
    """Fix 4.2: Zero-inductance inductor should raise ValueError."""
    try:
        from components.inductor import Inductor
        l = Inductor("L_BAD", {"type": "L", "n1": 1, "n2": 0, "value": 0.0})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "0 H" in str(e)

test("Zero-inductance inductor raises ValueError (Fix 4.2)", test_inductor_zero_value)

def test_negative_capacitor():
    """Fix 4.2: Negative capacitance should raise ValueError."""
    try:
        from components.capacitor import Capacitor
        c = Capacitor("C_BAD", {"type": "C", "n1": 1, "n2": 0, "value": -1e-6})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "negative" in str(e).lower()

test("Negative capacitance raises ValueError (Fix 4.2)", test_negative_capacitor)

def test_opamp_is_linear():
    """Fix 1.5: OpAmp should have IS_NONLINEAR = False."""
    from components.opamp import OpAmp
    assert OpAmp.IS_NONLINEAR == False, "OpAmp should be linear"

test("OpAmp IS_NONLINEAR is False (Fix 1.5)", test_opamp_is_linear)

def test_node_keys_discovery():
    """Fix 6.2 / 1.3: Circuit should discover nodes via NODE_KEYS."""
    p = NetlistParser()
    path = "testfiles/_test_nk.txt"
    with open(path, "w") as f:
        # VCVS with output on node 3, inputs on 2 and 0
        f.write("V1 1 0 DC 1\nR1 1 2 10k\nR2 2 0 10k\nE1 3 0 2 0 100000\nR3 3 0 1MEG\n.OP\n")
    comps, analyses = p.parse(path)
    circuit = Circuit(comps)
    # Node 3 (the OpAmp output) must be in the node map
    assert 3 in circuit.node_map, f"Node 3 not found. Map: {circuit.node_map}"
    os.remove(path)

test("OpAmp n_out node discovered via NODE_KEYS (Fix 1.3/6.2)", test_node_keys_discovery)

def test_voltage_source_sensitivity():
    """Fix 3.1: VoltageSource should have get_sensitivities."""
    from components.voltage_source import VoltageSource
    vs = VoltageSource("V1", {"type": "V", "n1": 1, "n2": 0, "value": 5.0})
    # Just verify method exists and returns non-empty dict
    node_map = {1: 0, "V1": 1}
    vs.bind_nodes(node_map)
    VI = np.array([5.0, 0.01])  # dummy
    Psi = np.array([0.5, 0.3])  # dummy
    sens = vs.get_sensitivities(VI=VI, PsiPhi=Psi)
    assert "V1" in sens, f"Expected 'V1' in sensitivities, got: {sens}"

test("VoltageSource has get_sensitivities (Fix 3.1)", test_voltage_source_sensitivity)

# ======================================================================
# 3. DIODE OVERFLOW TEST
# ======================================================================
print("\n" + "="*60)
print("3. NUMERICAL ROBUSTNESS TESTS")
print("="*60)

def test_diode_overflow_safe():
    """Fix 4.5: Diode model should not overflow for large Vd."""
    import core.models as models
    # Test extreme forward bias
    result = models.evaluate_diode(100.0, 1e-14, 0.02585)
    assert np.isfinite(result["I_D"]), f"I_D is not finite: {result['I_D']}"
    assert np.isfinite(result["gd"]), f"gd is not finite: {result['gd']}"
    assert result["I_D"] > 0, "Forward current should be positive"
    
    # Test extreme reverse bias
    result2 = models.evaluate_diode(-100.0, 1e-14, 0.02585)
    assert np.isfinite(result2["I_D"]), f"I_D reverse not finite: {result2['I_D']}"
    assert result2["I_D"] < 0, "Reverse current should be negative (≈ -Is)"

test("Diode model handles extreme voltages (Fix 4.5)", test_diode_overflow_safe)

def test_mosfet_model_renamed():
    """Fix 2.5: Model function should be evaluate_mosfet_level1."""
    import core.models as models
    assert hasattr(models, 'evaluate_mosfet_level1'), "evaluate_mosfet_level1 not found"
    # Ensure it works
    result = models.evaluate_mosfet_level1(2.0, 3.0, 0.7, 1e-3)
    assert result["I_D"] > 0

test("MOSFET model function renamed correctly (Fix 2.5)", test_mosfet_model_renamed)

# ======================================================================
# 4. FULL SIMULATION TESTS
# ======================================================================
print("\n" + "="*60)
print("4. FULL SIMULATION PIPELINE TESTS")
print("="*60)

def test_op_simulation():
    """Full .OP pipeline: voltage divider should give V(2) = 5V."""
    circuit, result = run_simulation_core("testfiles/voltage_divider.txt", sensitivity=True)
    assert result is not None
    assert result.type == ".OP"
    v2 = result.get_voltage(2)
    assert abs(v2 - 5.0) < 0.001, f"Expected V(2)=5.0, got {v2}"
    # Check that sensitivity data exists
    params = result.get_sensitivity_parameters(2)
    assert len(params) > 0, "Expected sensitivity parameters"
    print(f"    V(2) = {v2:.4f}V, sens params: {params}")

test(".OP simulation (voltage divider)", test_op_simulation)

def test_transient_simulation():
    """Full .TRAN pipeline: RC lowpass with pulse input."""
    circuit, result = run_simulation_core("testfiles/rc_lowpass.txt")
    assert result is not None
    assert result.type == ".TRAN"
    assert len(result.sweep_axis) > 10
    v_out = result.get_voltage(2)
    assert len(v_out) == len(result.sweep_axis)
    # At t=0, output should be 0V (from DC OP)
    assert abs(v_out[0]) < 0.01, f"Expected V(2,t=0)≈0, got {v_out[0]}"
    # At some later time, output should have charged
    max_v = np.max(v_out)
    assert max_v > 1.0, f"RC should charge above 1V, max was {max_v}"
    print(f"    {len(result.sweep_axis)} steps, V(2) range: [{np.min(v_out):.3f}, {max_v:.3f}]")

test(".TRAN simulation (RC lowpass)", test_transient_simulation)

def test_ac_simulation():
    """Full .AC pipeline: RC lowpass bode plot."""
    circuit, result = run_simulation_core("testfiles/rc_ac.txt", sensitivity=True)
    assert result is not None
    assert result.type == ".AC"
    v_out = result.get_voltage(2)
    assert np.iscomplexobj(v_out), "AC results should be complex"
    
    # Check magnitude at DC-ish (1 Hz): should be close to 0 dB
    mag_low = 20 * np.log10(np.abs(v_out[0]))
    assert mag_low > -1.0, f"Low-freq gain should be ~0dB, got {mag_low:.2f}dB"
    
    # Check magnitude at high freq: should be well below 0 dB
    mag_high = 20 * np.log10(np.abs(v_out[-1]))
    assert mag_high < -20.0, f"High-freq gain should be < -20dB, got {mag_high:.2f}dB"
    print(f"    Gain at 1Hz: {mag_low:.1f}dB, Gain at 100kHz: {mag_high:.1f}dB")

test(".AC simulation (RC lowpass bode)", test_ac_simulation)

def test_dc_sweep_simulation():
    """Fix 1.1: Full .DC pipeline must actually work now."""
    circuit, result = run_simulation_core("testfiles/dc_sweep.txt", sensitivity=True)
    assert result is not None
    assert result.type == ".DC", f"Expected .DC, got {result.type}"
    v_out = result.get_voltage(2)
    sweep = result.sweep_axis
    
    # For a 1:1 divider, V(2) should be V1/2
    for i in range(len(sweep)):
        expected = sweep[i] / 2.0
        actual = v_out[i]
        assert abs(actual - expected) < 0.01, f"At V1={sweep[i]}, expected V(2)={expected}, got {actual}"
    print(f"    {len(sweep)} sweep points, V(2) tracks V1/2 correctly")

test(".DC sweep simulation (Fix 1.1 end-to-end)", test_dc_sweep_simulation)

def test_cmos_inverter():
    """Nonlinear test: CMOS inverter with both NMOS and PMOS."""
    circuit, result = run_simulation_core("testfiles/cmos_inverter.txt", sensitivity=True)
    assert result is not None
    assert result.type == ".OP"
    
    v_out = result.get_voltage("out")
    v_vdd = result.get_voltage("vdd")
    
    # With Vin = 2.5V (mid-rail), Vout should be near mid-rail too
    assert 0.5 < v_out < 4.5, f"Inverter output unexpected: V(out)={v_out:.3f}"
    assert abs(v_vdd - 5.0) < 0.01, f"VDD should be 5V, got {v_vdd}"
    
    # Check sensitivity data
    params = result.get_sensitivity_parameters("out")
    assert any("VTO" in p for p in params), f"Expected VTO sensitivity, got: {params}"
    print(f"    V(out) = {v_out:.4f}V, V(vdd) = {v_vdd:.4f}V")
    print(f"    Sensitivity params: {params}")

test("CMOS inverter (.OP nonlinear)", test_cmos_inverter)

def test_opamp_circuit():
    """Fix 1.3/1.4: OpAmp in a circuit should work end-to-end."""
    path = "testfiles/_test_opamp_sim.txt"
    with open(path, "w") as f:
        # Non-inverting amplifier: Gain = 1 + R2/R1 = 1 + 10k/10k = 2
        f.write("* Non-inverting amplifier\n")
        f.write("V1 1 0 DC 1.0\n")
        f.write("R1 2 0 10k\n")
        f.write("R2 2 3 10k\n")
        f.write("E1 3 0 1 2 100000\n")  
        f.write("RL 3 0 1MEG\n")
        f.write(".OP\n")
    
    circuit, result = run_simulation_core(path, sensitivity=True)
    assert result is not None
    assert result.type == ".OP"
    
    v_out = result.get_voltage(3)
    # With gain ~100k and feedback, output should be close to 2V
    assert abs(v_out - 2.0) < 0.1, f"Expected V(3)≈2.0V, got {v_out:.4f}"
    print(f"    Non-inverting amp: V(in)=1.0V, V(out)={v_out:.4f}V (gain≈{v_out/1.0:.2f})")
    os.remove(path)

test("OpAmp non-inverting amplifier (Fix 1.3/1.4 end-to-end)", test_opamp_circuit)

# ======================================================================
# 5. SENSITIVITY CONSISTENCY TESTS
# ======================================================================
print("\n" + "="*60)
print("5. SENSITIVITY CONSISTENCY TESTS")
print("="*60)

def test_op_sensitivity_finite():
    """All .OP sensitivities should be finite real numbers."""
    circuit, result = run_simulation_core(
        "testfiles/voltage_divider.txt", 
        output_nodes=[2],
        sensitivity=True
    )
    params = result.get_sensitivity_parameters(2)
    for p in params:
        val = result.get_sensitivity(2, p)
        # val may be an array of length 1 for .OP
        s = val[0] if hasattr(val, '__len__') else val
        assert np.isfinite(s), f"Sensitivity d(2)/d({p}) = {s} is not finite"
    print(f"    All {len(params)} sensitivities are finite")

test(".OP sensitivities are all finite", test_op_sensitivity_finite)

def test_dc_sweep_sensitivity():
    """DC sweep sensitivities should be arrays matching sweep length."""
    circuit, result = run_simulation_core(
        "testfiles/dc_sweep.txt",
        output_nodes=[2],
        sensitivity=True
    )
    params = result.get_sensitivity_parameters(2)
    assert len(params) > 0, "Expected sensitivity params for DC sweep"
    for p in params:
        val = result.get_sensitivity(2, p)
        assert len(val) == len(result.sweep_axis), \
            f"Sensitivity array length mismatch for {p}: {len(val)} vs {len(result.sweep_axis)}"
        assert np.all(np.isfinite(val)), f"Non-finite values in sensitivity for {p}"
    print(f"    {len(params)} params, all arrays length {len(result.sweep_axis)}, all finite")

test(".DC sweep sensitivities match sweep length", test_dc_sweep_sensitivity)

def test_ac_sensitivity():
    """AC sensitivities should be complex arrays matching frequency length."""
    circuit, result = run_simulation_core(
        "testfiles/rc_ac.txt",
        output_nodes=[2],
        sensitivity=True
    )
    params = result.get_sensitivity_parameters(2)
    assert len(params) > 0, "Expected sensitivity params for AC"
    for p in params:
        val = result.get_sensitivity(2, p)
        assert len(val) == len(result.sweep_axis), \
            f"Length mismatch for {p}: {len(val)} vs {len(result.sweep_axis)}"
        assert np.all(np.isfinite(np.abs(val))), f"Non-finite magnitude in sens for {p}"
    print(f"    {len(params)} params, all complex arrays, all finite")

test(".AC sensitivities are valid complex arrays", test_ac_sensitivity)

def test_transient_sensitivity():
    """Transient sensitivity in all output formats."""
    circuit, result = run_simulation_core(
        "testfiles/rc_lowpass.txt",
        output_nodes=[2],
        sensitivity=True
    )
    params = result.get_sensitivity_parameters(2)
    assert len(params) > 0
    
    for p in params:
        # Integrated (scalar)
        integ = result.get_sensitivity(2, p, output_format="integrated")
        assert np.isfinite(integ), f"Integrated sens for {p} not finite: {integ}"
        
        # Series (array)
        series = result.get_sensitivity(2, p, output_format="series")
        assert len(series) == len(result.sweep_axis)
        
        # Accumulator (array)
        accum = result.get_sensitivity(2, p, output_format="accumulator")
        assert len(accum) == len(result.sweep_axis)
    
    print(f"    {len(params)} params, all 3 output formats valid")

test(".TRAN sensitivities in all formats", test_transient_sensitivity)

# ======================================================================
# 6. PLOTTING (just verify no crashes)
# ======================================================================
print("\n" + "="*60)
print("6. PLOTTING TESTS (smoke tests)")
print("="*60)

def test_plot_transient():
    from utils.plotting import plot_transient
    circuit, result = run_simulation_core("testfiles/rc_lowpass.txt")
    plot_transient(result, output_nodes=2, folder="figures/tran", name="test_tran")
    assert os.path.exists("figures/tran/test_tran.png")

test("Transient plot saves without crash", test_plot_transient)

def test_plot_bode():
    from utils.plotting import make_bode_plot
    circuit, result = run_simulation_core("testfiles/rc_ac.txt")
    make_bode_plot(result, output_nodes=2, folder="figures/ac", name="test_bode")
    assert os.path.exists("figures/ac/test_bode.png")

test("Bode plot saves without crash", test_plot_bode)

def test_plot_dc_sweep():
    from utils.plotting import plot_dc_sweep
    circuit, result = run_simulation_core("testfiles/dc_sweep.txt")
    plot_dc_sweep(result, output_nodes=2, folder="figures/dc", name="test_dc")
    assert os.path.exists("figures/dc/test_dc.png")

test("DC sweep plot saves without crash", test_plot_dc_sweep)

# ======================================================================
# 7. TRAPEZOIDAL RULE TESTS
# ======================================================================
print("\n" + "="*60)
print("7. TRAPEZOIDAL RULE INTEGRATION TESTS")
print("="*60)

def test_tr_transient_basic():
    """TR transient should produce valid results for RC lowpass."""
    circuit, result = run_simulation_core("testfiles/rc_lowpass.txt", method='TR')
    assert result is not None
    assert result.type == ".TRAN"
    v_out = result.get_voltage(2)
    assert abs(v_out[0]) < 0.01, f"Expected V(2,t=0)≈0, got {v_out[0]}"
    max_v = np.max(v_out)
    assert max_v > 1.0, f"RC should charge above 1V with TR, max was {max_v}"
    print(f"    TR: {len(result.sweep_axis)} steps, V(2) range: [{np.min(v_out):.3f}, {max_v:.3f}]")

test(".TRAN with TR method (RC lowpass)", test_tr_transient_basic)

def test_tr_vs_be_accuracy():
    """TR should be more accurate than BE for the same time step.
    
    For an RC circuit with τ = RC = 1kΩ × 1μF = 1ms:
    At t=1ms (one time constant), V_out should be (1 - e^{-1}) × 5V ≈ 3.16V.
    TR (second-order) should be closer to the analytical answer than BE (first-order).
    """
    # Run both methods
    _, result_be = run_simulation_core("testfiles/rc_lowpass.txt", method='BE')
    _, result_tr = run_simulation_core("testfiles/rc_lowpass.txt", method='TR')
    
    v_be = result_be.get_voltage(2)
    v_tr = result_tr.get_voltage(2)
    time = result_be.sweep_axis
    
    # Find index closest to t = 1ms (one time constant)
    tau_idx = np.argmin(np.abs(time - 1e-3))
    
    # Analytical: V = 5 * (1 - exp(-t/RC)) during the first pulse half
    # RC = 1k * 1u = 1ms, so at t=1ms: V = 5*(1-e^{-1}) ≈ 3.1606
    v_analytical = 5.0 * (1.0 - np.exp(-1.0))
    
    err_be = abs(v_be[tau_idx] - v_analytical)
    err_tr = abs(v_tr[tau_idx] - v_analytical)
    
    print(f"    At t=1ms (τ): analytical={v_analytical:.4f}V")
    print(f"    BE: V={v_be[tau_idx]:.4f}V, error={err_be:.6f}V")
    print(f"    TR: V={v_tr[tau_idx]:.4f}V, error={err_tr:.6f}V")
    
    # TR should be at least as accurate as BE (typically much better)
    assert err_tr <= err_be * 1.1, \
        f"TR error ({err_tr:.6e}) should be ≤ BE error ({err_be:.6e})"

test("TR more accurate than BE at same step size", test_tr_vs_be_accuracy)

def test_tr_with_sensitivity():
    """TR transient with full sensitivity pass."""
    circuit, result = run_simulation_core(
        "testfiles/rc_lowpass.txt", output_nodes=[2], sensitivity=True, method='TR'
    )
    params = result.get_sensitivity_parameters(2)
    assert len(params) > 0, f"Expected sensitivity parameters with TR, got none"
    
    for p in params:
        integ = result.get_sensitivity(2, p, output_format="integrated")
        assert np.isfinite(integ), f"TR integrated sens for {p} not finite: {integ}"
        
        series = result.get_sensitivity(2, p, output_format="series")
        assert len(series) == len(result.sweep_axis)
        assert np.all(np.isfinite(series)), f"TR series sens for {p} has non-finite values"
    
    print(f"    TR sensitivity: {len(params)} params, all finite")

test(".TRAN TR with sensitivity pass", test_tr_with_sensitivity)

def test_tr_sensitivity_vs_be():
    """TR and BE sensitivities should agree in sign and approximate magnitude."""
    _, result_be = run_simulation_core(
        "testfiles/rc_lowpass.txt", output_nodes=[2], sensitivity=True, method='BE'
    )
    _, result_tr = run_simulation_core(
        "testfiles/rc_lowpass.txt", output_nodes=[2], sensitivity=True, method='TR'
    )
    
    params_be = result_be.get_sensitivity_parameters(2)
    params_tr = result_tr.get_sensitivity_parameters(2)
    
    # Same parameters should be available
    assert set(params_be) == set(params_tr), \
        f"BE params {params_be} ≠ TR params {params_tr}"
    
    for p in params_be:
        s_be = result_be.get_sensitivity(2, p, output_format="integrated")
        s_tr = result_tr.get_sensitivity(2, p, output_format="integrated")
        
        # Same sign
        if abs(s_be) > 1e-12:
            assert (s_be > 0) == (s_tr > 0), \
                f"Sign mismatch for {p}: BE={s_be:.4e}, TR={s_tr:.4e}"
        
        # Within 50% (they use different integration methods, so won't be identical)
        if abs(s_be) > 1e-10:
            ratio = abs(s_tr / s_be)
            assert 0.5 < ratio < 2.0, \
                f"Magnitude divergence for {p}: BE={s_be:.4e}, TR={s_tr:.4e}, ratio={ratio:.2f}"
    
    print(f"    BE vs TR sensitivities agree in sign and ~magnitude for {len(params_be)} params")

test("TR vs BE sensitivity consistency", test_tr_sensitivity_vs_be)

def test_tr_nonlinear():
    """TR should work with nonlinear circuits (CMOS inverter transient)."""
    path = "testfiles/_test_cmos_tran.txt"
    with open(path, "w") as f:
        f.write("* CMOS Inverter Transient TR Test\n")
        f.write("VDD vdd 0 DC 5\n")
        f.write("VIN in 0 PULSE(0 5 0 1n 1n 5u 10u)\n")
        f.write("M1 out in vdd vdd PMOD W=10u L=1u\n")
        f.write("M2 out in 0 0 NMOD W=10u L=1u\n")
        f.write("CL out 0 1p\n")
        f.write(".MODEL NMOD NMOS (VTO=0.7 KP=120u)\n")
        f.write(".MODEL PMOD PMOS (VTO=0.7 KP=60u)\n")
        f.write(".TRAN 0.1u 20u\n")
    
    circuit, result = run_simulation_core(path, method='TR')
    assert result is not None
    assert result.type == ".TRAN"
    
    v_out = result.get_voltage("out")
    # Output should swing between near-0 and near-5V
    assert np.min(v_out) < 1.0, f"Inverter output min too high: {np.min(v_out):.2f}"
    assert np.max(v_out) > 4.0, f"Inverter output max too low: {np.max(v_out):.2f}"
    print(f"    CMOS TR: V(out) range [{np.min(v_out):.2f}, {np.max(v_out):.2f}]")
    os.remove(path)

test("TR with nonlinear CMOS inverter transient", test_tr_nonlinear)

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "="*60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
print("="*60)

if FAIL > 0:
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
