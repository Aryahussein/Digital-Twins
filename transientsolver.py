from assembleYmatrix import get_idx

def stamp_capacitor_transient(Y, sources, n1, n2, C, node_map, deltaT, cap_voltage):
    """
    Stamps a capacitor into the admittance matrix Y using the Back Euler approximation
    Y: NxN sparse lilmatrix (complex)
    n1, n2: node numbers
    C: capacitance (Farads)
    w: angular frequency (rad/s)
    node_map: dict mapping node numbers to matrix indices
    """

    # Capacitor's Geq (equivalent admittance) for Backwards Euler; stamps into the "Y" matrix
    
    Geq = C/deltaT # inverse of R = deltaT/2C
    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    
    if i is not None:
        Y[i, i] += Geq
        if j is not None:
            Y[i, j] -= Geq
            Y[j, i] -= Geq
    if j is not None:
        Y[j, j] += Geq


    # Capacitor's Ieq (equivalent current source) for Backwards Euler; stamps into the "sources" matrix
    Ieq = C*cap_voltage/deltaT

    i, j = get_idx(n1, node_map), get_idx(n2, node_map)
    if i is not None: sources[i] -= Ieq
    if j is not None: sources[j] += Ieq

