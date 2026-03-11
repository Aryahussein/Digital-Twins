from components import Resistor, Capacitor, VoltageSource, Mosfet # Import Inductor, Diode, etc. later

def create_component(name, data_dict):
    """Factory function to spawn the correct object based on type."""
    type_char = data_dict["type"]
    
    if type_char == 'R': return Resistor(name, data_dict)
    if type_char == 'C': return Capacitor(name, data_dict)
    if type_char == 'V': return VoltageSource(name, data_dict)
    if type_char == 'M': return Mosfet(name, data_dict)
    # Add 'L', 'I', 'G', 'D' as you build out those subclasses
    
    raise ValueError(f"Unknown component type in Netlist: {type_char} for {name}")
