import numpy as np
from assembleYmatrix import initialize_stamps, stamp_mna_connections, stamp_static_components, stamp_source_components, stamp_dynamic_components
from solver import solve_nonlinear_circuit, solve_linear_circuit

class DCEngine:
    def __init__(self, components, node_map, is_complex, is_nonlinear, ramp):
        self.components = components
        self.node_map = node_map
        self.is_complex = is_complex
        self.is_nonlinear = is_nonlinear
        self.ramp = ramp
        self.total_dim = len(node_map)

    def build_base_matrices(self):
        """Builds the static topology that never changes."""
        Y_base, sources_base = initialize_stamps(self.total_dim, is_complex=self.is_complex)
        stamp_mna_connections(Y_base, self.components, self.node_map)
        stamp_static_components(Y_base, sources_base, self.components, self.node_map)
        return Y_base.tocsc(), sources_base

    def compute_dc_bias(self, Y_base, sources_base, evaluated_components=None):
        """Finds the t=0 operating point (Caps=Open, Inds=Short)."""
        comps = evaluated_components if evaluated_components is not None else self.components
        
        Y_dc = Y_base.copy()
        sources_dc = sources_base.copy()
        
        stamp_source_components(Y_dc, sources_dc, comps, self.node_map)
        stamp_dynamic_components(Y_dc, sources_dc, comps, self.node_map, w=0.0)
        
        if self.is_nonlinear:
            return solve_nonlinear_circuit(
                Y_dc, sources_dc, comps, self.node_map, 
                np.zeros_like(sources_dc), max_iter=100, num_steps=self.ramp
            )
        return solve_linear_circuit(Y_dc, sources_dc)
