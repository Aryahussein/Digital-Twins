from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_nonlinear_circuit, solve_linear_circuit, solve_adjoint
from assembleYmatrix import generate_stamps
import numpy as np
from tools import run_bode_plot, print_solution, get_all_sensitivities, plot_sensitivity_sweep
from constants import *

def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    variance = 0
    for name, sens in sensitivities.items():
        sigma_p = components[name]["value"] * percent_sigma
        variance += np.abs(sens * sigma_p)**2
    return np.sqrt(variance)

def do_sensitivity_analysis(lu, VI, output_node, node_map, w=0.0):
    PsiPhi = solve_adjoint(lu, output_node, node_map)
    sensitivities = get_all_sensitivities(components, VI, PsiPhi, node_map, w=w)
    tolerance_on_components = 0.01
    std_dev = estimate_std_dev(sensitivities, components, percent_sigma=tolerance_on_components)
    return sensitivities, std_dev

if __name__ == "__main__":
    test_directory = "testfiles/"
    netlist = test_directory + "/test_opamp_buffer.txt"

    #-----------------------------------------------------------------------------------
    # Build circuit
    #-----------------------------------------------------------------------------------
    components, analyses = parse_netlist(netlist)
    print(components)

    nonlinear = False
    for comp in components.values():
        ctype = str(comp.get("type", "")).upper()
        if ctype in ("D", "A"):
            nonlinear = True
            break

    node_map = build_node_index(components)
    total_dim = len(node_map)

    # PUT THE FREQUENCY SOMEWHERE ELSE!!
    w = 2*np.pi * 60
    if nonlinear:
        w = 0.0

    Y, sources = generate_stamps(components, node_map, total_dim, w=w)

    #-----------------------------------------------------------------------------------
    # solve circuit
    #-----------------------------------------------------------------------------------
    if nonlinear:
        V_guess = np.zeros(total_dim)
        max_iter = 100
        tol = 1e-9
        num_ramp_steps = 10
        lu, VI = solve_nonlinear_circuit(
            Y, sources, components, node_map, V_guess,
            max_iter=max_iter, tol=tol, num_steps=num_ramp_steps
        )
    else:
        lu, VI = solve_linear_circuit(Y, sources)

    print_solution(VI, node_map, w=w)
    # -----------------------------------------------------------------------------------
# solve circuit
# -----------------------------------------------------------------------------------
if nonlinear:
    V_guess = np.zeros(total_dim)
    max_iter = 100
    tol = 1e-9
    num_ramp_steps = 10

    lu, VI = solve_nonlinear_circuit(
        Y, sources, components, node_map, V_guess,
        max_iter=max_iter, tol=tol, num_steps=num_ramp_steps
    )
else:
    lu, VI = solve_linear_circuit(Y, sources)

print_solution(VI, node_map, w=w)

# ---------------------------------------------------------
# Sensitivity of BRANCH CURRENT through A1
# ---------------------------------------------------------

output_target_for_sensitivity = "A1"   # <-- this is the key change

sensitivities, std_dev = do_sensitivity_analysis(
    lu,
    VI,
    output_target_for_sensitivity,   # pass "A1" instead of node number
    node_map,
    w=w
)

print("\n--- Sensitivities ---")
for name, val in sensitivities.items():
    print(f"{name:15s}: {val}")

print(f"\nEstimated std deviation: {std_dev}")








