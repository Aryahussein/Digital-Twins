import numpy as np
from scipy.sparse import lil_matrix
from constants import *
import component_stamps as stamps

# 1. Purely static components (frequency/time independent, stamped once)
STATIC_DISPATCH = {
    'R': stamps.stamp_resistor,
    'G': stamps.stamp_vccs,
}

# 2. Source components (need re-stamping in transient for time-varying values)
SOURCE_DISPATCH = {
    'V': stamps.stamp_independent_voltage,
    'I': stamps.stamp_current_source,
}

# 3. Dynamic components (frequency-dependent, for AC analysis)
DYNAMIC_DISPATCH = {
    'L': stamps.stamp_inductor,
    'C': stamps.stamp_capacitor,
}

# 4. Transient companion models (Backward Euler discretization)
TRANSIENT_DISPATCH = {
    'C': stamps.stamp_capacitor_be,
    'L': stamps.stamp_inductor_be,
}

# 5. Non-linear components (re-stamped each Newton-Raphson iteration)
NONLINEAR_DISPATCH = {
    'D': stamps.stamp_diode,
}

# 5. Non-linear components (re-stamped each Newton-Raphson iteration)
NONLINEAR_DISPATCH = {
    'D': stamps.stamp_diode,
    'M': stamps.stamp_mosfet, # Add this line
}


def initialize_stamps(total_dim, w=0):
    """Create empty sparse matrix and source vector with appropriate dtype."""
    dtype = float if w == 0 else complex
    Y = lil_matrix((total_dim, total_dim), dtype=dtype)
    sources = np.zeros(total_dim, dtype=dtype)
    return Y, sources


def stamp_static_components(Y, sources, components, node_map):
    """
    Stamps only frequency/time-independent components (R, G) into the matrix.
    These form the truly static base matrix that never changes between steps.
    Returns CSC matrix for efficient solving.
    """
    for name, comp in components.items():
        type_char = name[0].upper()
        if type_char in STATIC_DISPATCH:
            STATIC_DISPATCH[type_char](Y, sources, comp, node_map, name)

    return Y.tocsc(), sources


def stamp_source_components(Y, sources, components, node_map):
    """
    Stamps independent sources (V, I) into the matrix and source vector.
    Separated from static stamps so transient analysis can re-stamp sources
    per time step without double-counting.
    """
    for name, comp in components.items():
        type_char = name[0].upper()
        if type_char in SOURCE_DISPATCH:
            SOURCE_DISPATCH[type_char](Y, sources, comp, node_map, name)

    return Y, sources


def stamp_linear_components(Y, sources, components, node_map):
    """
    Stamps all linear static components (R, G, V, I) into the matrix.
    Use this for OP and AC analysis where sources don't change.
    Returns CSC matrix for efficient solving.
    """
    for name, comp in components.items():
        type_char = name[0].upper()
        if type_char in STATIC_DISPATCH:
            STATIC_DISPATCH[type_char](Y, sources, comp, node_map, name)
        elif type_char in SOURCE_DISPATCH:
            SOURCE_DISPATCH[type_char](Y, sources, comp, node_map, name)

    return Y.tocsc(), sources


def stamp_dynamic_components(Y, sources, components, node_map, w=0.0):
    """
    Stamps frequency-dependent components (C, L) for AC analysis at angular frequency w.
    Modifies Y and sources in-place.
    """
    for name, comp in components.items():
        type_char = name[0].upper()
        if type_char in DYNAMIC_DISPATCH:
            DYNAMIC_DISPATCH[type_char](Y, sources, comp, node_map, name, w=w)

    return Y.tocsc(), sources


def stamp_transient_components(Y, sources, components, node_map, dt, v_prev):
    """
    Stamps sources and Backward Euler companion models for a transient time step.
    Sources are re-stamped here because their values may change over time.
    Companion models (C, L) use the previous solution as history terms.
    Modifies Y and sources in-place.
    """
    for name, comp in components.items():
        type_char = name[0].upper()

        if type_char in SOURCE_DISPATCH:
            SOURCE_DISPATCH[type_char](Y, sources, comp, node_map, name)

        if type_char in TRANSIENT_DISPATCH:
            TRANSIENT_DISPATCH[type_char](Y, sources, comp, node_map, name, dt=dt, v_prev=v_prev)

    return Y.tocsc(), sources


def stamp_nonlinear_components(Y, sources, components, node_map, v_prev, v_guess):
    """
    Stamps linearized non-linear components for Newton-Raphson iteration.
    Modifies Y and sources in-place.
    """
    for name, comp in components.items():
        type_char = name[0].upper()
        if type_char in NONLINEAR_DISPATCH:
            NONLINEAR_DISPATCH[type_char](Y, sources, comp, node_map, v_prev, v_guess)

    return Y.tocsc(), sources
