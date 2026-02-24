import numpy as np
from scipy.sparse import lil_matrix
import component_stamps as stamps

# =============================================================================
# DISPATCH TABLES
# =============================================================================

LINEAR_DISPATCH = {
    'R': stamps.stamp_resistor,
    'G': stamps.stamp_vccs,
    'V': stamps.stamp_independent_voltage,
    'I': stamps.stamp_current_source,
}

DYNAMIC_DISPATCH = {
    'C': stamps.stamp_capacitor,
    'L': stamps.stamp_inductor,
}

NONLINEAR_DISPATCH = {
    'D': stamps.stamp_diode,
    'A': stamps.stamp_opamp_tanh,
}

# =============================================================================
# MAIN GENERATE FUNCTION (this was missing)
# =============================================================================

def generate_stamps(components, node_map, total_dim, w=0.0):
    """
    Build base matrix Y and source vector.
    Nonlinear devices are NOT stamped here.
    """
    dtype = float if w == 0 else complex
    Y = lil_matrix((total_dim, total_dim), dtype=dtype)
    sources = np.zeros(total_dim, dtype=dtype)

    for name, comp in components.items():
        ctype = comp.get("type", "").upper()

        # Linear elements
        if ctype in LINEAR_DISPATCH:
            LINEAR_DISPATCH[ctype](Y, sources, comp, node_map, name)

        # AC dynamic elements
        if w != 0 and ctype in DYNAMIC_DISPATCH:
            DYNAMIC_DISPATCH[ctype](Y, sources, comp, node_map, name, w=w)

    return Y.tocsc(), sources


# =============================================================================
# NONLINEAR STAMPING (called during Newton iterations)
# =============================================================================

def stamp_nonlinear_components(Y, sources, components, node_map, p_V_guess, V_guess):
    """
    Stamp nonlinear devices into matrix for current Newton iteration.
    """
    for name, comp in components.items():
        ctype = comp.get("type", "").upper()

        if ctype in NONLINEAR_DISPATCH:
            NONLINEAR_DISPATCH[ctype](
                Y, sources, comp, node_map, p_V_guess, V_guess
            )

    return Y.tocsc(), sources