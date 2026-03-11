import numpy as np
from sources import evaluate_all_time_sources
from assembleYmatrix import stamp_transient_components
from solver import solve_nonlinear_circuit, solve_linear_circuit

class TransientEngine:
    def __init__(self, components, node_map, is_nonlinear, ramp):
        self.components = components
        self.node_map = node_map
        self.is_nonlinear = is_nonlinear
        self.ramp = ramp
        self.total_dim = len(node_map)

    def _solve_single_step(self, Y_base, sources_base, comp_t, dt, v_prev):
        Y_step = Y_base.copy()
        sources_step = sources_base.copy()
        stamp_transient_components(Y_step, sources_step, comp_t, self.node_map, dt, v_prev)
        
        if self.is_nonlinear:
            return solve_nonlinear_circuit(Y_step, sources_step, comp_t, self.node_map, v_prev, max_iter=100, num_steps=self.ramp, print_stuff=False)
        return solve_linear_circuit(Y_step, sources_step)

    def run(self, Y_base, sources_base, v_initial, t_stop, dt, keep_lus=False):
        """Steps forward through time."""
        time_array = np.arange(0, t_stop, dt)
        results = np.zeros((len(time_array), self.total_dim))
        list_of_lus = []
        
        v_prev = v_initial
        
        for step, t in enumerate(time_array):
            if step % max(1, len(time_array)//10) == 0: 
                print(f"Solving forward time {t:.3f}")
                
            comp_t = evaluate_all_time_sources(self.components, t)
            lu, VI = self._solve_single_step(Y_base, sources_base, comp_t, dt, v_prev)
            
            results[step, :] = VI
            v_prev = VI
            if keep_lus: list_of_lus.append(lu)
            
        return time_array, results, list_of_lus
