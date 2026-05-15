class TemporaryCircuitState:
    """Safely mutates a circuit and guarantees restoration upon exit."""
    
    def __init__(self, circuit, param_map, shifts):
        self.circuit = circuit
        self.param_map = param_map
        self.shifts = shifts  # Dictionary of { "R1_value": 1100, "M1_W": 2e-6 }
        self.original_state = {}

    def __enter__(self):
        # 1. Backup original values and apply the new shifts
        for param, new_val in self.shifts.items():
            comp = self.param_map[param]
            self.original_state[param] = comp.get_nominal_value(param)
            comp.set_nominal_value(param, new_val)
            
        self.circuit.clear_cache()
        return self.circuit

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 2. Automatically restore the pristine state, EVEN IF AN ERROR HAPPENED
        for param, orig_val in self.original_state.items():
            comp = self.param_map[param]
            comp.set_nominal_value(param, orig_val)
            
        self.circuit.clear_cache()
        # Return False to let any exceptions bubble up naturally
        return False
