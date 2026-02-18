import numpy as np
from scipy.sparse import lil_matrix
import component_stamps as stamps

# 1. Static components
LINEAR_DISPATCH = {
    'R': stamps.stamp_resistor,
    'G': stamps.stamp_vccs,
    'V': stamps.stamp_independent_voltage,
    'I': stamps.stamp_current_source
}

# 2. Dynamic components
TRANSIENT_DISPATCH = {
    'C': stamps.stamp_capacitor_be,
    'L': stamps.stamp_inductor_be
}

# 3. Non-linear components
NONLINEAR_DISPATCH = {
    'D': stamps.stamp_diode_linearized
}

def generate_basis_matrix(components, node_map, total_dim, dt=None, v_prev=None, sources_prev=None, mode="TRAN"):
    """
    Builds the 'Base Matrix' for the time step.
    If mode == "DC", Capacitors are Open and Inductors are Short.
    """
    Y = lil_matrix((total_dim, total_dim), dtype=float)
    sources = np.zeros(total_dim, dtype=float)

    for name, comp in components.items():
        type_char = name[0].upper()
        
        # --- DC ANALYSIS MODE ---
        if mode == "DC":
            # Ignore Capacitors (Open Circuit)
            if type_char == 'C':
                pass 
            # Treat Inductors as Resistors with very low resistance (Short)
            elif type_char == 'L':
                short_comp = comp.copy()
                short_comp["value"] = 1e-9 # 1 nano-ohm
                stamps.stamp_resistor(Y, sources, short_comp, node_map)
            # Stamp everything else normally
            elif type_char in LINEAR_DISPATCH:
                LINEAR_DISPATCH[type_char](Y, sources, comp, node_map, name=name)

        # --- TRANSIENT ANALYSIS MODE ---
        else:
            if type_char in LINEAR_DISPATCH:
                LINEAR_DISPATCH[type_char](Y, sources, comp, node_map, name=name)
            elif type_char in TRANSIENT_DISPATCH:
                TRANSIENT_DISPATCH[type_char](Y, sources, comp, node_map, name=name, dt=dt, v_prev=v_prev, sources_prev=sources_prev)

    return Y, sources

def stamp_nonlinear_components(Y, sources, components, node_map, v_guess):
    """
    Stamps only the non-linear components into an EXISTING matrix Y.
    Modifies Y and sources in-place.
    """
    for name, comp in components.items():
        type_char = name[0].upper()
        if type_char in NONLINEAR_DISPATCH:
            NONLINEAR_DISPATCH[type_char](Y, sources, comp, node_map, name=name, v_guess=v_guess)