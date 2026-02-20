import numpy as np
from scipy.sparse import lil_matrix
from constants import *
import component_stamps as stamps

# 1. Static components
LINEAR_DISPATCH = {
    'R': stamps.stamp_resistor,
    'G': stamps.stamp_vccs,
    'V': stamps.stamp_independent_voltage,
    'I': stamps.stamp_current_source,
}

SOURCE_DISPATCH = {
    'V': stamps.stamp_independent_voltage,
    'I': stamps.stamp_current_source
}

# 2. Dynamic components
TRANSIENT_DISPATCH = {
    'C': stamps.stamp_capacitor_be,
    'L': stamps.stamp_inductor_be
}

DYNAMIC_DISPATCH = {
    'L': stamps.stamp_inductor,
    'C': stamps.stamp_capacitor
}

# 3. Non-linear components
NONLINEAR_DISPATCH = {
    'D': stamps.stamp_diode
}

def initialize_stamps(total_dim, w=0):
    dtype = float if w==0 else complex
    Y = lil_matrix((total_dim, total_dim), dtype=dtype)
    sources = np.zeros(total_dim, dtype=dtype)
    return Y, sources

def stamp_linear_components(Y, sources, components, node_map):
    """
    Builds the 'Base Matrix' for the time step.
    If mode == "DC", Capacitors are Open and Inductors are Short.
    """
    for name, comp in components.items():
        type_char = name[0].upper()

        if type_char in LINEAR_DISPATCH:
            LINEAR_DISPATCH[type_char](Y, sources, comp, node_map, name)

    return Y.tocsc(), sources


def stamp_dynamic_components(Y, sources, components, node_map, w=0.0):
    """
    Stamps only the transient components (energy storing components) into an EXISTING matrix Y.
    Modifies Y and sources in-place.
    """
    for name, comp in components.items():
        type_char = name[0].upper()

        if type_char in DYNAMIC_DISPATCH:
            DYNAMIC_DISPATCH[type_char](Y, sources, comp, node_map, name, w=w)

    return Y.tocsc(), sources

def stamp_transient_components(Y, sources, components, node_map, dt, v_prev):
    """
    Stamps only the transient components (energy storing components) into an EXISTING matrix Y.
    Modifies Y and sources in-place.
    """
    for name, comp in components.items():
        type_char = name[0].upper()

        if type_char in SOURCE_DISPATCH:
            # print(f"stamping source {name}")
            SOURCE_DISPATCH[type_char](Y, sources, comp, node_map, name)

        if type_char in TRANSIENT_DISPATCH:
            TRANSIENT_DISPATCH[type_char](Y, sources, comp, node_map, name, dt=dt, v_prev=v_prev)

    # print("sources", sources)
    return Y.tocsc(), sources

def stamp_nonlinear_components(Y, sources, components, node_map, v_prev, v_guess):
    """
    Stamps only the non-linear components into an EXISTING matrix Y.
    Modifies Y and sources in-place.
    """
    for name, comp in components.items():
        type_char = name[0].upper()
        if type_char in NONLINEAR_DISPATCH:
            NONLINEAR_DISPATCH[type_char](Y, sources, comp, node_map, v_prev, v_guess)

    return Y.tocsc(), sources

