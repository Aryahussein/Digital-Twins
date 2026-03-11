from dc_engine import DCEngine
from transient_engine import TransientEngine
from ac_engine import ACEngine
from adjoint_engine import AdjointEngine
from results import SimulationResult
from sources import evaluate_all_time_sources

class Simulator:
    def __init__(self, components, analyses, node_map, output_nodes=None, ramp=1):
        self.components = components
        self.analyses = analyses
        self.node_map = node_map
        self.output_nodes = output_nodes if output_nodes else list(node_map.keys())
        self.ramp = ramp
        
        self.is_nonlinear = any(comp["type"] in ['D', 'M'] for comp in components.values())
        self.is_complex = ".AC" in analyses or (".OP" in analyses and analyses[".OP"].get("freq", 0.0) > 0.0)

    def execute_analysis(self, sensitivity=False, keep_lus=False):
        # 1. Start the Foundation (DCEngine)
        dc_engine = DCEngine(self.components, self.node_map, self.is_complex, self.is_nonlinear, self.ramp)
        Y_base, sources_base = dc_engine.build_base_matrices()

        # -----------------------------------------------------------------
        # TRANSIENT ROUTE
        # -----------------------------------------------------------------
        if ".TRAN" in self.analyses:
            print("Running transient analysis...")
            t_stop, dt = self.analyses[".TRAN"]["stop"], self.analyses[".TRAN"]["step"]
            
            # Get t=0 starting point
            comp_t0 = evaluate_all_time_sources(self.components, 0.0)
            _, v_initial = dc_engine.compute_dc_bias(Y_base, sources_base, evaluated_components=comp_t0)
            
            # Run Forward Engine
            tran_engine = TransientEngine(self.components, self.node_map, self.is_nonlinear, self.ramp)
            time, VI, lus = tran_engine.run(Y_base, sources_base, v_initial, t_stop, dt, keep_lus=(keep_lus or sensitivity))
            
            # Package Vault
            result = SimulationResult(".TRAN", time, VI, self.node_map, dt=dt)
            
            # Run Backward Engine
            if sensitivity:
                adj_engine = AdjointEngine(self.components, self.node_map, self.output_nodes)
                result.sensitivities = adj_engine.compute_transient(time, VI, lus, dt)
                
            return result

        # -----------------------------------------------------------------
        # AC ROUTE
        # -----------------------------------------------------------------
        elif ".AC" in self.analyses:
            print("Running AC analysis...")
            start, stop, pts = self.analyses[".AC"]["start"], self.analyses[".AC"]["stop"], self.analyses[".AC"]["num_points"]
            
            # Get DC linearization point
            _, v_dc = dc_engine.compute_dc_bias(Y_base, sources_base)
            
            # Run AC Engine
            ac_engine = ACEngine(self.components, self.node_map, self.is_nonlinear)
            freq, VI, lus = ac_engine.run(Y_base, v_dc, start, stop, pts, keep_lus)
            
            result = SimulationResult(".AC", freq, VI, self.node_map)
            return result
