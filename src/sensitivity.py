import numpy as np
from constants import *
import models
from assembleYmatrix import build_adjoint_history_source, update_adjoint_state


def get_all_sensitivities(components, VI, PsiPhi, node_map, w=0.0, dt=None, V_DC=None, V_prev=None):
    """
    Compute sensitivity of the output w.r.t. all component parameters using
    the adjoint vector PsiPhi.
    
    For AC analysis, pass w (angular frequency).
    For transient analysis, pass dt (time step) — this uses BE companion model derivatives.
    
    Supported: R, C, L, G, V (source value), I (source value)
    Not yet supported: D (diode Is sensitivity)
    """
    sensitivities = {}

    op_voltages = V_DC if V_DC is not None else VI

    for name, comp in components.items():
        # 1. Get Indices and branch voltages
        n1, n2 = comp.get("n1", 0), comp.get("n2", 0)
        idx1, idx2 = node_map.get(n1), node_map.get(n2)

        # Helper to get the voltage difference across a branch
        # (v1 - v2). If a node is ground, its voltage is 0.
        VI_branch = (VI[idx1] if idx1 is not None else 0) - \
                   (VI[idx2] if idx2 is not None else 0)
        
        PsiPhi_branch = (PsiPhi[idx1] if idx1 is not None else 0) - \
                       (PsiPhi[idx2] if idx2 is not None else 0)

        # 2. Apply the sensitivity formula based on component type
        if name.startswith("R"):
            R = comp["value"]
            sensitivities[name] = (1.0 / (R**2)) * (VI_branch * PsiPhi_branch)

        elif name.startswith("C"):
            if dt is not None and V_prev is not None:
                # True Transient Adjoint: Sensitivity depends on dV/dt (Current!)
                v_prev_diff = (V_prev[idx1] if idx1 is not None else 0.0) - \
                              (V_prev[idx2] if idx2 is not None else 0.0)
                
                dV_dt = (VI_branch - v_prev_diff) / dt
                sensitivities[name] = -PsiPhi_branch * dV_dt
            else:
                # AC Analysis
                sensitivities[name] = -1j * w * (VI_branch * PsiPhi_branch)

        elif name.startswith("L"):
            l_curr_idx = node_map[name]
            i_L = VI[l_curr_idx]
            i_L_hat = PsiPhi[l_curr_idx]
            
            if dt is not None and V_prev is not None:
                # True Transient Adjoint: Sensitivity depends on dI/dt (Voltage!)
                i_L_prev = V_prev[l_curr_idx]
                dI_dt = (i_L - i_L_prev) / dt
                sensitivities[name] = i_L_hat * dI_dt
            else:
                # AC Analysis
                sensitivities[name] = 1j * w * (i_L * i_L_hat)


        elif name.startswith("G"):
            n3, n4 = comp.get("n3", 0), comp.get("n4", 0)
            idx3, idx4 = node_map.get(n3), node_map.get(n4)
            
            v_sense = (VI[idx3] if idx3 is not None else 0) - \
                      (VI[idx4] if idx4 is not None else 0)
            
            sensitivities[name] = - (PsiPhi_branch * v_sense)

        elif name.startswith("V"):
            v_branch_idx = node_map[name]
            sensitivities[name] = PsiPhi[v_branch_idx]

        elif name.startswith("I"):
            sensitivities[name] = -PsiPhi_branch

        elif name.startswith("D"):
            # 1. Parameter Extraction
            if "value" in comp:
                Is = comp.get("value")
            elif "model" in comp:
                Is = comp["model_params"]["IS"]
            else:
                Is = 1e-14 # Fallback
            
            # 2. Operating Point Voltage Extraction
            vd_op = (op_voltages[idx1] if idx1 is not None else 0.0) - \
                    (op_voltages[idx2] if idx2 is not None else 0.0)
            
            # 3. Model Evaluation
            diode_data = models.evaluate_diode(vd_op, Is, Vt)
            
            # 4. Sensitivity Calculation
            sensitivities[f"{name}_Is"] = -PsiPhi_branch * diode_data["dId_dIs"]

        elif name.startswith("M"):
            # 1. Parameter Extraction (Matching your stamper exactly)
            n_d, n_g, n_s = comp.get("n_d"), comp.get("n_g"), comp.get("n_s")
            idx_d, idx_g, idx_s = node_map.get(n_d), node_map.get(n_g), node_map.get(n_s)
            
            params = comp.get("model_params", {})
            inst_params = comp.get("inst_params", {})
            
            VTO = params.get("VTO", 0.7)
            W = inst_params.get("W", 1e-6)
            L = inst_params.get("L", 1e-6)
            
            mu = params.get("MU", 0.0)
            Cox = params.get("C_OX", 0.0)
            
            if "KP" in params:
                KP = params["KP"]
            elif "MU" in params and "C_OX" in params:
                KP = mu * Cox
            else:
                raise ValueError(f"No KP or (MU and C_ox) specified for {name}!")
            
            Bn = (W / L) * KP
            
            # 2. Operating Point Voltage Extraction
            v_d = op_voltages[idx_d] if idx_d is not None else 0.0
            v_g = op_voltages[idx_g] if idx_g is not None else 0.0
            v_s = op_voltages[idx_s] if idx_s is not None else 0.0
            
            vgs_op = v_g - v_s
            vds_op = v_d - v_s
            
            # Adjoint voltage across Drain-Source
            Psi_DS = (PsiPhi[idx_d] if idx_d is not None else 0.0) - \
                     (PsiPhi[idx_s] if idx_s is not None else 0.0)

            # 3. Model Evaluation
            nmos_data = models.evaluate_nmos(vgs_op, vds_op, VTO, Bn)
            I_D = nmos_data["I_D"]
            
            # 4. Sensitivity Calculations
            if W != 0: sensitivities[f"{name}_W"] = -Psi_DS * (I_D / W)
            if L != 0: sensitivities[f"{name}_L"] =  Psi_DS * (I_D / L)
            
            # Check if KP was derived from MU and C_OX, or provided directly
            if "MU" in params and "C_OX" in params and mu != 0 and Cox != 0:
                sensitivities[f"{name}_Cox"] = -Psi_DS * (I_D / Cox)
                sensitivities[f"{name}_mu"]  = -Psi_DS * (I_D / mu)
            elif "KP" in params and KP != 0:
                sensitivities[f"{name}_KP"]  = -Psi_DS * (I_D / KP)

    return sensitivities

def compute_step_sensitivities(lu, VI, components, node_map, output_nodes=None, w=0.0, dt=None):
    """Solves the adjoint system and gathers sensitivities for requested output nodes.
    
    For AC analysis, pass w (angular frequency).
    For transient analysis, pass dt (time step).
    """
    from solver import solve_adjoint
    if output_nodes is None:
        output_nodes = list(node_map.keys())

    step_sensitivities = {}
    for out_node in output_nodes:
        PsiPhi = solve_adjoint(lu, out_node, node_map)
        comp_sens = get_all_sensitivities(components, VI, PsiPhi, node_map, w=w, dt=dt)
        step_sensitivities[out_node] = comp_sens

    return step_sensitivities

def aggregate_sweep_sensitivities(components, node_map, analyses, 
                                  output_nodes=None, raw_sensitivities=None, 
                                  list_of_lus=None, VI_list=None, freq_list=None, dt=None):
    """Aggregates multi-step simulation data into traced arrays.
    
    For post-processing mode (raw_sensitivities=None), pass freq_list for AC
    or dt for transient analysis.
    """
    print("Starting sensitivity data aggregation...")
    target_nodes = output_nodes if output_nodes is not None else list(node_map.keys())
    
    # Initialize nested dict: {node: {comp_name: []}}
    sensitivity_dict = {
        node: {name: [] for name in components.keys()} for node in target_nodes
    }

    # If raw sensitivities weren't computed during the simulation loop, compute them now
    if raw_sensitivities is None:
        if not list_of_lus or VI_list is None:
            return sensitivity_dict
            
        print("Computing sensitivities from stored LU matrices...")
        is_ac = ".AC" in analyses
        raw_sensitivities = []
        
        for i in range(len(list_of_lus)):
            if is_ac and freq_list is not None:
                w_step = 2 * np.pi * freq_list[i]
                step_sens = compute_step_sensitivities(
                    list_of_lus[i], VI_list[i], components, node_map, target_nodes, w=w_step
                )
            else:
                # Transient mode: pass dt
                step_sens = compute_step_sensitivities(
                    list_of_lus[i], VI_list[i], components, node_map, target_nodes, dt=dt
                )
            raw_sensitivities.append(step_sens)

    # Transpose data: raw_sensitivities is a list of dicts -> [{node: {comp: val}}]
    for step_data in raw_sensitivities:
        for node in target_nodes:
            for comp_name, sens_value in step_data[node].items():
                sensitivity_dict[node][comp_name].append(sens_value)

    # Convert lists to numpy arrays for easier plotting/math later
    for node in sensitivity_dict:
        for comp in sensitivity_dict[node]:
            sensitivity_dict[node][comp] = np.array(sensitivity_dict[node][comp])

    print("Sensitivity aggregation complete.")
    # print(sensitivity_dict)
    return sensitivity_dict

def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    """
    Computes output standard deviation.
    Works for scalars (Single Step) OR numpy arrays (Sweeps).
    """
    variance = 0.0
    for name, sens in sensitivities.items():
        # Skip empty arrays
        if isinstance(sens, np.ndarray) and sens.size == 0:
            continue
        # Skip zero scalar sensitivities
        if np.isscalar(sens) and sens == 0:
            continue

        # Safely get component value, default to 0 if it doesn't have a standard "value"
        comp_val = components[name].get("value", 0.0) 
        sigma_p = comp_val * percent_sigma
        
        # Add to variance (np.abs handles both reals and complex AC magnitudes safely)
        variance += np.abs(sens * sigma_p)**2
        
    return np.sqrt(variance)

def compute_transient_adjoint(components, node_map, time_array, V_forward, list_of_lus, dt, output_nodes, method='BE'):
    """
    Performs the backward time-traveling adjoint pass to compute integrated transient sensitivities.
    """
    total_dim = len(node_map)
    
    # --- THIS LINE WAS MISSING! ---
    num_steps = len(time_array) 
    
    v_hat_next = np.zeros(total_dim)
    adjoint_history = []
    
    # Initialize the persistent state memory (crucial for Trapezoidal later)
    adjoint_state = {} 

    print(f"\n--- Starting Backward Adjoint Pass ({method}) ---")
    for i in reversed(range(num_steps)):
        
        # 1. Build Adjoint RHS (History from Capacitors/Inductors)
        J_adjoint = build_adjoint_history_source(
            components, node_map, dt, v_hat_next, adjoint_state, method=method
        )

        # 2. Add Objective Excitation (Impulse at the FINAL time step)
        if i == num_steps - 1 and output_nodes:
            for node in output_nodes:
                idx = node_map.get(node)
                if idx is not None:
                    J_adjoint[idx] += 1.0

        # 3. Solve Backward Step
        lu = list_of_lus[i]
        v_hat = lu.solve(J_adjoint, trans='T')
        adjoint_history.insert(0, v_hat) # Insert at front to reverse the reverse

        # 4. Update the persistent state for the NEXT backward step
        update_adjoint_state(
            components, node_map, dt, v_hat_next, v_hat, adjoint_state, method=method
        )

        v_hat_next = v_hat

    print("--- Integrating Sensitivities over Time ---")
    total_sens = {}
    
    # We need a primary node key for your plotter's nested dictionary
    primary_node = output_nodes[0] if output_nodes else list(node_map.keys())[0]
    time_series_sens = {primary_node: {}}
    

    for i in range(num_steps):
        vi_step = V_forward[i]
        v_hat_step = adjoint_history[i]

        # GET PREVIOUS VOLTAGE FOR dV/dt MATH
        vi_prev = V_forward[i-1] if i > 0 else vi_step 

        # Pass V_prev into the sensitivity calculator!
        step_sens = get_all_sensitivities(
            components, VI=vi_step, PsiPhi=v_hat_step, 
            node_map=node_map, dt=dt, V_DC=vi_step, V_prev=vi_prev
        )

        for param, val in step_sens.items():
            # 1. Add to the integrated total
            total_sens[param] = total_sens.get(param, 0.0) + (val * dt)
            
            # 2. Append to the time-series array
            if param not in time_series_sens[primary_node]:
                time_series_sens[primary_node][param] = []
            time_series_sens[primary_node][param].append(val)

    # Convert all the lists into numpy arrays for the plotter
    for param in time_series_sens[primary_node]:
        time_series_sens[primary_node][param] = np.array(time_series_sens[primary_node][param])

    # Return BOTH the integrated scalars and the time-series arrays
    return {
        "Integrated_Transient": total_sens,
        "Time_Series": time_series_sens
    }
