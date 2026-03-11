import numpy as np
from sources import build_ac_sources
from assembleYmatrix import stamp_dynamic_components, stamp_nonlinear_components
from solver import solve_linear_circuit

class ACEngine:
    def __init__(self, components, node_map, is_nonlinear):
        self.components = components
        self.node_map = node_map
        self.is_nonlinear = is_nonlinear
        
    def _solve_single_point(self, w, Y_base, VI_dc, ac_sources):
        Y_ac = Y_base.astype(complex)
        sources_step = ac_sources.copy()
        
        stamp_dynamic_components(Y_ac, sources_step, self.components, self.node_map, w=w)
        
        if self.is_nonlinear:
            dummy_dc = np.zeros_like(sources_step)
            stamp_nonlinear_components(Y_ac, dummy_dc, self.components, self.node_map, v_prev=VI_dc, v_guess=VI_dc)
            
        return solve_linear_circuit(Y_ac, sources_step)

    def run(self, Y_base, VI_dc, start_freq, stop_freq, points, keep_lus=False):
        frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), points)
        VIs, list_of_lus = [], []
        ac_sources = build_ac_sources(self.components, self.node_map)
        
        for f in frequencies:
            w = 2 * np.pi * f
            lu_ac, VI_ac = self._solve_single_point(w, Y_base, VI_dc, ac_sources)
            VIs.append(VI_ac)
            if keep_lus: list_of_lus.append(lu_ac)
            
        return frequencies, np.array(VIs), list_of_lus
