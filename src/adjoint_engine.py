import numpy as np
from constants import *
import models
from assembleYmatrix import build_adjoint_history_source, update_adjoint_state
from solver import solve_adjoint

class AdjointEngine:
    def __init__(self, circuit, output_nodes=None):
        self.circuit = circuit  # Store the whole circuit object!
        self.output_nodes = output_nodes if output_nodes else list(circuit.node_map.keys())

    def _get_all_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_DC=None, V_prev=None):
        sensitivities = {}
        op_voltages = V_DC if V_DC is not None else VI

        for name, comp in self.circuit.components.items():
            # Use object attributes instead of dictionary lookups
            n1, n2 = comp.get_node("n1"), comp.get_node("n2")
            idx1, idx2 = self.circuit.get_idx(n1), self.circuit.get_idx(n2)

            VI_branch = (VI[idx1] if idx1 is not None else 0) - \
                        (VI[idx2] if idx2 is not None else 0)
            PsiPhi_branch = (PsiPhi[idx1] if idx1 is not None else 0) - \
                            (PsiPhi[idx2] if idx2 is not None else 0)

            if name.startswith("R"):
                R = comp.value
                sensitivities[name] = (1.0 / (R**2)) * (VI_branch * PsiPhi_branch)

            elif name.startswith("C"):
                if dt is not None and V_prev is not None:
                    v_prev_diff = (V_prev[idx1] if idx1 is not None else 0.0) - \
                                  (V_prev[idx2] if idx2 is not None else 0.0)
                    dV_dt = (VI_branch - v_prev_diff) / dt
                    sensitivities[name] = -PsiPhi_branch * dV_dt
                else:
                    sensitivities[name] = -1j * w * (VI_branch * PsiPhi_branch)

            elif name.startswith("L"):
                l_curr_idx = self.circuit.get_idx(name)
                i_L = VI[l_curr_idx]
                i_L_hat = PsiPhi[l_curr_idx]
                
                if dt is not None and V_prev is not None:
                    i_L_prev = V_prev[l_curr_idx]
                    dI_dt = (i_L - i_L_prev) / dt
                    sensitivities[name] = i_L_hat * dI_dt
                else:
                    sensitivities[name] = 1j * w * (i_L * i_L_hat)

            # ... [Keep G, V, I logic updated with comp.get_node() and comp.value] ...

            elif name.startswith("M"):
                n_d, n_g, n_s = comp.get_node("n_d"), comp.get_node("n_g"), comp.get_node("n_s")
                idx_d, idx_g, idx_s = self.circuit.get_idx(n_d), self.circuit.get_idx(n_g), self.circuit.get_idx(n_s)
                
                VTO = comp.model_params.get("VTO", 0.7)
                W = comp.inst_params.get("W", 1e-6)
                L = comp.inst_params.get("L", 1e-6)
                
                mu = comp.model_params.get("MU", 0.0)
                Cox = comp.model_params.get("C_OX", 0.0)
                KP = comp.model_params.get("KP", mu * Cox if mu and Cox else None)
                
                Bn = (W / L) * KP
                v_d = op_voltages[idx_d] if idx_d is not None else 0.0
                v_g = op_voltages[idx_g] if idx_g is not None else 0.0
                v_s = op_voltages[idx_s] if idx_s is not None else 0.0
                
                vgs_op, vds_op = v_g - v_s, v_d - v_s
                Psi_DS = (PsiPhi[idx_d] if idx_d is not None else 0.0) - \
                         (PsiPhi[idx_s] if idx_s is not None else 0.0)

                nmos_data = models.evaluate_nmos(vgs_op, vds_op, VTO, Bn)
                I_D = nmos_data["I_D"]
                
                if W != 0: sensitivities[f"{name}_W"] = -Psi_DS * (I_D / W)
                if L != 0: sensitivities[f"{name}_L"] =  Psi_DS * (I_D / L)
                
        return sensitivities

    def compute_transient(self, time_array, V_forward, list_of_lus, dt, method='BE'):
        num_steps = len(time_array)
        v_hat_next = np.zeros(self.circuit.total_dim)
        adjoint_history, adjoint_state = [], {}

        print(f"\n--- Starting Backward Adjoint Pass ({method}) ---")
        for i in reversed(range(num_steps)):
            # Pass the circuit to your stamper!
            J_adjoint = build_adjoint_history_source(
                self.circuit.components, self.circuit.node_map, dt, v_hat_next, adjoint_state, method=method
            )

            if i == num_steps - 1:
                for node in self.output_nodes:
                    idx = self.circuit.get_idx(node)
                    if idx is not None: J_adjoint[idx] += 1.0

            lu = list_of_lus[i]
            v_hat = lu.solve(J_adjoint, trans='T')
            adjoint_history.insert(0, v_hat)

            update_adjoint_state(
                self.circuit.components, self.circuit.node_map, dt, v_hat_next, v_hat, adjoint_state, method=method
            )
            v_hat_next = v_hat

        print("--- Integrating Sensitivities over Time ---")
        total_sens, primary_node = {}, self.output_nodes[0]
        time_series_sens = {primary_node: {}}

        for i in range(num_steps):
            vi_step, v_hat_step = V_forward[i], adjoint_history[i]
            vi_prev = V_forward[i-1] if i > 0 else vi_step 

            step_sens = self._get_all_sensitivities(
                VI=vi_step, PsiPhi=v_hat_step, dt=dt, V_DC=vi_step, V_prev=vi_prev
            )

            for param, val in step_sens.items():
                total_sens[param] = total_sens.get(param, 0.0) + (val * dt)
                if param not in time_series_sens[primary_node]:
                    time_series_sens[primary_node][param] = []
                time_series_sens[primary_node][param].append(val)

        for param in time_series_sens[primary_node]:
            time_series_sens[primary_node][param] = np.array(time_series_sens[primary_node][param])

        return {"Integrated_Transient": total_sens, "Time_Series": time_series_sens}
