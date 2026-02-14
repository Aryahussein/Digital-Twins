from assembleYmatrix import get_idx

def stamp_capacitor_transient(Y_temp, sources, n1, n2, C, node_map, deltaT, cap_voltage):
    """
    Stamps a capacitor into the admittance matrix Y using the trapezoidal approximation for Geq = 2C/(deltaT)
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    C: capacitance (Farads)
    w: angular frequency (rad/s)
    node_map: dict mapping node numbers to matrix indices
    """

    #### Stamp g into Y matrix ####
    g = C/deltaT # inverse of R = deltaT/2C         ### Note - change to Trapezoidal - currently Backwards Euler
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y_temp[i, i] += g
        if j is not None:
            Y_temp[i, j] -= g
            Y_temp[j, i] -= g
    if j is not None:
        Y_temp[j, j] += g


    ####  Backwards Euler for the current source
    I_value = C*cap_voltage/deltaT

    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    if i is not None: sources[i] -= I_value
    if j is not None: sources[j] += I_value

