# run_tests.py
import os
import numpy as np

from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_nonlinear_circuit, solve_linear_circuit
from assembleYmatrix import generate_stamps
from tools import print_solution


def pick_test_directory() -> str:
    """
    Your main.py uses testfiles/. But your uploaded netlists are currently in the repo root.
    This chooses whichever exists.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "testfiles")
    return candidate if os.path.isdir(candidate) else here


def is_nonlinear(components: dict) -> bool:
    # Current code treats diodes as nonlinear
    return any(name.startswith("D") for name in components.keys())


def choose_omega(netlist_name: str, nonlinear: bool) -> float:
    """
    For nonlinear tests -> DC only (w=0).
    For AC tests -> solve one spot frequency so we at least exercise complex stamping.
    """
    if nonlinear:
        return 0.0

    lname = netlist_name.lower()
    if "ac_" in lname or "resonance" in lname or "lowpass" in lname:
        f = 1_000.0  # 1 kHz spot check
        return 2 * np.pi * f

    return 0.0  # DC for everything else


def solve_netlist(path: str):
    components = parse_netlist(path)
    node_map, total_dim = build_node_index(components)

    nonlinear = is_nonlinear(components)
    w = choose_omega(os.path.basename(path), nonlinear)

    Y, sources = generate_stamps(components, node_map, total_dim, w=w)

    if nonlinear:
        V_guess = np.zeros(total_dim, dtype=float)
        lu, VI = solve_nonlinear_circuit(
            Y, sources, components, node_map, total_dim,
            V_guess, max_iter=100, tol=1e-9, num_steps=10
        )
    else:
        lu, VI = solve_linear_circuit(Y, sources)

    return components, node_map, total_dim, w, VI


def voltage_at(node, VI, node_map):
    # node is an int node label used in the netlist (0 is ground)
    if node == 0:
        return 0.0 + 0.0j
    return VI[node_map[node]]


def check_voltage_sources(components, VI, node_map, tol=1e-6):
    """
    Universal sanity check:
    For every independent voltage source Vx n1 n2 value:
        V(n1) - V(n2) should equal value (phasor, so value is real here).
    This is true for DC and AC phasor solves.
    """
    for name, comp in components.items():
        if not name.startswith("V"):
            continue

        n1 = comp["n1"]
        n2 = comp["n2"]
        value = comp["value"]

        vdrop = voltage_at(n1, VI, node_map) - voltage_at(n2, VI, node_map)
        err = abs(vdrop - value)

        if err > tol:
            raise AssertionError(
                f"{name}: expected V({n1})-V({n2})={value}, got {vdrop} (abs err {err})"
            )


def run_one(netlist_path: str, verbose=False):
    components, node_map, total_dim, w, VI = solve_netlist(netlist_path)

    # Print solution if you want to see it
    if verbose:
        print_solution(VI, node_map, w=w)

    # Minimal but powerful check
    check_voltage_sources(components, VI, node_map, tol=1e-6)

    # Extra sanity: no NaNs/Infs
    if np.any(~np.isfinite(np.real(VI))) or np.any(~np.isfinite(np.imag(VI))):
        raise AssertionError("Solution contains NaN or Inf.")

    return True


def main():
    test_dir = pick_test_directory()

    # Add/remove tests here
    tests = [
        "ac_lowpass.txt",
        "ac_resonance.txt",
        "test_diode.txt",
        "test_lots_of_diodes.txt",
        "test_with_vccs.txt",
        "test_complex_mna.txt",
        # You can include these too (they may be DC-only):
        "series_resistors.txt",
        "parallel_resistors.txt",
        "example_lecture.txt",
        "large_test.txt",
    ]

    print(f"Using test directory: {test_dir}")
    passed = 0
    failed = 0

    for t in tests:
        path = os.path.join(test_dir, t)
        if not os.path.exists(path):
            print(f"[SKIP] {t} (not found at {path})")
            continue

        try:
            run_one(path, verbose=False)
            print(f"[PASS] {t}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t}")
            print(f"       {e}")
            failed += 1

    print("\n=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    # Make VS Code show a non-zero exit code if failures happen
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()