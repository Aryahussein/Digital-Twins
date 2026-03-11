"""
Component Factory Module.

This module implements the Factory Method design pattern to instantiate
specific polymorphic component objects based on their SPICE type character.
"""

from core.components import (
    Resistor, 
    Capacitor, 
    VoltageSource, 
    Mosfet,
    Inductor,
    CurrentSource,
    Diode,
    VCCS,
    OpAmp
)

# Extension: A registry map provides O(1) routing and makes adding new 
# components as simple as adding one line to this dictionary.
_COMPONENT_REGISTRY = {
    'R': Resistor,
    'C': Capacitor,
    'V': VoltageSource,
    'M': Mosfet,
    'L': Inductor,
    'I': CurrentSource,
    'D': Diode,
    'G': VCCS,
    'E': OpAmp
}

def create_component(name, data_dict):
    """Instantiates the correct Component subclass based on the netlist type.

    Reads the 'type' field from the parsed data dictionary and routes the data
    to the appropriate class constructor defined in the component registry.

    Args:
        name (str): The unique netlist name of the component (e.g., 'R1', 'M_MAIN').
        data_dict (dict): The parsed parameter dictionary containing at minimum
            a 'type' key (e.g., {'type': 'R', 'n1': '1', 'n2': '0', 'value': 1000}).

    Returns:
        Component: An instantiated object inheriting from core.components.Component.

    Raises:
        KeyError: If the 'type' key is entirely missing from the data_dict.
        ValueError: If the component type is not recognized by the registry.
    """
    type_char = data_dict.get("type")
    
    if not type_char:
        raise KeyError(f"Component '{name}' is missing a 'type' attribute in the parsed netlist.")

    # Extension: Ensure the type character is uppercase for safety (e.g., 'r' -> 'R')
    type_char = str(type_char).upper()

    # Look up the class blueprint in the registry
    component_class = _COMPONENT_REGISTRY.get(type_char)
    
    if component_class is None:
        raise ValueError(
            f"Unknown component type '{type_char}' for '{name}'. "
            f"Supported types are: {list(_COMPONENT_REGISTRY.keys())}"
        )
        
    # Instantiate and return the object
    return component_class(name, data_dict)
