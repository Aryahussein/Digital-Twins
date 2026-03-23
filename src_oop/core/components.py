"""
Circuit Components Module.

This module defines the base Component interface and implements the specific
polymorphic behaviors (matrix stamping, sensitivity calculations) for various
electrical components (Resistors, Capacitors, Sources, Transistors).
"""

import numpy as np
import core.models as models # Contains diode/mosfet equations (evaluate_nmos)
from core.waveforms import Waveform

class Component:
    """The Base Class defining the interface for all circuit components.

    Attributes:
        name (str): The unique netlist name (e.g., 'R1', 'M_INV').
        type (str): The component type character (e.g., 'R', 'C').
        value (float): The default value of the component, if applicable.
        data (dict): The raw parsed dictionary from the netlist.
        idx_1 (int or None): The primary matrix index (often positive terminal).
        idx_2 (int or None): The secondary matrix index (often negative terminal).
    """
    def __init__(self, name, data_dict):
        self.name = name
        self.type = data_dict.get("type")
        self.value = data_dict.get("value", 0.0)
        self.data = data_dict 
        
        self.idx_1 = None 
        self.idx_2 = None

    def bind_nodes(self, node_map):
        """Translates string/integer node names into matrix indices.

        Args:
            node_map (dict): The global mapping of node names to matrix indices.
        """
        pass 

    # === POLYMORPHIC METHODS ===
    def stamp_mna_connection(self, Y):
        """Stamps MNA branch topology (+1/-1) into the admittance matrix."""
        pass
        
    def stamp_static(self, Y):
        """Stamps time/frequency-independent linear terms (e.g., Conductance)."""
        pass
        
    def stamp_dc(self, Y, sources):
        """Stamps DC values for Operating Point (.OP) and initial transient steps."""
        pass
        
    def stamp_ac(self, Y, sources, w):
        """Stamps frequency-dependent complex impedances and AC phasors.
        
        Args:
            w (float): Angular frequency in rad/s.
        """
        pass
        
    def stamp_transient(self, Y, sources, t, dt, v_prev): 
        """Stamps dynamic companion models (Backward Euler) and time-varying sources.
        
        Args:
            t (float): Current simulation time in seconds.
            dt (float): Current time step size.
            v_prev (np.ndarray): The solution vector from the previous time step.
        """
        pass
        
    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps linearized conductances (gm, gds) and equivalent currents for NR.
        
        Args:
            p_V_guess (np.ndarray): The voltage solution from the previous iteration.
            V_guess (np.ndarray): The current working voltage guess.
        """
        pass
    
    # === ADJOINT METHODS ===
    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'):
        """Builds the RHS current history vector for the backward adjoint sweep."""
        pass
        
    def update_adjoint_state(self, dt, v_hat_next, v_hat, adjoint_state, method='BE'):
        """Updates internal memory states required by integration methods like TR."""
        pass
        
    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates the parameter sensitivities using the Adjoint method.
        
        Returns:
            dict: A mapping of parameter names (e.g., 'R1', 'M1_W') to scalar sensitivity values.
        """
        return {}

    def _stamp_branch_equation(self, Y):
        """Shared helper method for components that require MNA branch currents.
        
        Requires `self.branch_idx` to be initialized in `bind_nodes()`.
        """
        i, j, b = getattr(self, 'idx_1', None), getattr(self, 'idx_2', None), getattr(self, 'branch_idx', None)
        
        if b is None: return 
            
        if i is not None:
            Y[i, b] += 1
            Y[b, i] += 1
        if j is not None:
            Y[j, b] -= 1
            Y[b, j] -= 1

# =====================================================================
# LINEAR COMPONENTS
# =====================================================================
class Resistor(Component):
    """A linear resistor component."""
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def stamp_static(self, Y):
        g = 1.0 / self.value
        i, j = self.idx_1, self.idx_2
        if i is not None:
            Y[i, i] += g
            if j is not None:
                Y[i, j] -= g
                Y[j, i] -= g
        if j is not None:
            Y[j, j] += g

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        
        return {self.name: (1.0 / (self.value**2)) * ((v1 - v2) * (p1 - p2))}

class VCCS(Component):
    """Voltage-Controlled Current Source (Type 'G'). I = G * (V_pos - V_neg)."""

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0)) # Output +
        self.idx_2 = node_map.get(self.data.get("n2", 0)) # Output -
        self.idx_3 = node_map.get(self.data.get("n3", 0)) # Control +
        self.idx_4 = node_map.get(self.data.get("n4", 0)) # Control -

    def stamp_static(self, Y):
        g = self.value # The transconductance
        i, j, k, l = self.idx_1, self.idx_2, self.idx_3, self.idx_4
        
        # Current out of node i depends on (Vk - Vl)
        if i is not None:
            if k is not None: Y[i, k] += g
            if l is not None: Y[i, l] -= g
        
        # Current into node j depends on (Vk - Vl)
        if j is not None:
            if k is not None: Y[j, k] -= g
            if l is not None: Y[j, l] += g

    # AC and Transient inherit from static as the gain is frequency independent.
    # def stamp_ac(self, Y, sources, w): self.stamp_static(Y, sources)
    # def stamp_transient(self, Y, sources, t, dt, v_prev): self.stamp_static(Y, sources)

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """
        Calculates sensitivity w.r.t the Transconductance (G).
        Formula: -(Psi_out+ - Psi_out-) * (V_ctrl+ - V_ctrl-)
        """
        # 1. Get Forward (Primal) Control Voltage
        v3 = VI[self.idx_3] if self.idx_3 is not None else 0.0
        v4 = VI[self.idx_4] if self.idx_4 is not None else 0.0
        v_control = v3 - v4

        # 2. Get Backward (Adjoint) Output Voltage
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        psi_output = p1 - p2

        # 3. Sensitivity is the product of the control voltage and the adjoint output
        # Note the negative sign: it comes from the MNA matrix derivative
        sens_g = -(psi_output * v_control)

        return {self.name: sens_g}


class CurrentSource(Component):
    """An independent current source (Type 'I')."""
    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        
        # 1. Setup Transient Waveform (PULSE, SIN, etc.)
        self.waveform = None
        if "source" in data_dict:
            self.waveform = Waveform(data_dict["source"])
            
        # 2. Setup AC Phasor for Frequency Sweeps
        self.ac_mag = data_dict.get("ac_mag", 0.0)
        self.ac_phase = data_dict.get("ac_phase", 0.0)
        self.phasor = self.ac_mag * np.exp(1j * np.radians(self.ac_phase))

    def bind_nodes(self, node_map):
        """Maps terminals to matrix indices."""
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def stamp_dc(self, Y, sources):
        """Stamps the steady-state DC value (at t=0)."""
        val = self.waveform.get_value(0.0) if self.waveform else self.value
        self._apply_rhs_stamp(sources, val)

    def stamp_transient(self, Y, sources, t, dt, v_prev):
        """Stamps the time-varying value based on the waveform function."""
        val = self.waveform.get_value(t) if self.waveform else self.value
        self._apply_rhs_stamp(sources, val)

    def stamp_ac(self, Y, sources, w):
        """Stamps the complex AC phasor."""
        if self.ac_mag != 0.0:
            self._apply_rhs_stamp(sources, self.phasor)

    def _apply_rhs_stamp(self, sources, current_val):
        """Internal helper to apply the nodal current flow."""
        if self.idx_1 is not None:
            sources[self.idx_1] -= current_val
        if self.idx_2 is not None:
            sources[self.idx_2] += current_val

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Returns the sensitivity of the output w.r.t the source value."""
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        # For a current source, sensitivity is -(psi_i - psi_j)
        return {self.name: -(p1 - p2)}

class VoltageSource(Component):
    """An independent voltage source requiring an MNA branch equation."""
    def __init__(self, name, data_dict):
        super().__init__(name, data_dict)
        
        self.waveform = None
        if "source" in data_dict:
            self.waveform = Waveform(data_dict["source"])
            
        self.ac_mag = data_dict.get("ac_mag", 0.0)
        self.ac_phase = data_dict.get("ac_phase", 0.0)
        self.phasor = self.ac_mag * np.exp(1j * np.radians(self.ac_phase))

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name)

    def stamp_mna_connection(self, Y):
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        if self.waveform:
            sources[self.branch_idx] = self.waveform.get_value(0.0)
        else:
            sources[self.branch_idx] = self.value

    def stamp_transient(self, Y, sources, t, dt, v_prev):
        if self.waveform:
            current_volts = self.waveform.get_value(t)
        else:
            current_volts = self.value
            
        sources[self.branch_idx] = current_volts

    def stamp_ac(self, Y, sources, w):
        if self.ac_mag != 0.0:
            sources[self.branch_idx] += self.phasor

# =====================================================================
# ENERGY STORAGE COMPONENTS
# =====================================================================
class Capacitor(Component):
    """A dynamic capacitor component."""
    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))

    def stamp_ac(self, Y, sources, w):
        """EXTENSION ADDED: Stamps AC impedance."""
        g = 1j * w * self.value
        i, j = self.idx_1, self.idx_2
        if i is not None:
            Y[i, i] += g
            if j is not None:
                Y[i, j] -= g
                Y[j, i] -= g
        if j is not None:
            Y[j, j] += g

    def stamp_transient(self, Y, sources, t, dt, v_prev):
        g_eq = self.value / dt
        i, j = self.idx_1, self.idx_2

        v1_prev = v_prev[i] if i is not None else 0.0
        v2_prev = v_prev[j] if j is not None else 0.0
        I_eq = g_eq * (v1_prev - v2_prev)

        if i is not None: Y[i, i] += g_eq
        if j is not None: Y[j, j] += g_eq
        if i is not None and j is not None:
            Y[i, j] -= g_eq
            Y[j, i] -= g_eq

        if i is not None: sources[i] += I_eq
        if j is not None: sources[j] -= I_eq

    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'):
        i, j = self.idx_1, self.idx_2
        v1_hat = v_hat_next[i] if i is not None else 0.0
        v2_hat = v_hat_next[j] if j is not None else 0.0

        I_eq = (self.value / dt) * (v1_hat - v2_hat)
        if i is not None: J_hist[i] += I_eq
        if j is not None: J_hist[j] -= I_eq

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivity w.r.t Capacitance (C)."""
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0

        # Adjoint potential difference: (Psi_1 - Psi_2)
        adj_diff = (p1 - p2)

        # 1. AC Analysis
        if w != 0.0:
            # dI/dC = j * w * V_diff
            dI_dC = 1j * w * (v1 - v2)
            return {self.name: -adj_diff * dI_dC}

        # 2. Transient Analysis
        elif dt is not None and V_prev is not None:
            v1_prev = V_prev[self.idx_1] if self.idx_1 is not None else 0.0
            v2_prev = V_prev[self.idx_2] if self.idx_2 is not None else 0.0
            
            # dI/dC = dV/dt using Backward Euler
            dV_dt = ((v1 - v2) - (v1_prev - v2_prev)) / dt
            return {self.name: -adj_diff * dV_dt}

        # 3. DC Analysis
        else:
            # Capacitors are open circuits in DC; changing C has no effect
            return {self.name: 0.0}


class Inductor(Component):
    """A linear inductor (Type 'L'). Requires an MNA branch equation."""

    def bind_nodes(self, node_map):
        self.idx_1 = node_map.get(self.data.get("n1", 0))
        self.idx_2 = node_map.get(self.data.get("n2", 0))
        self.branch_idx = node_map.get(self.name) # Inductors get a branch index

    def stamp_mna_connection(self, Y):
        """Stamps the +1/-1 topology for the branch current."""
        self._stamp_branch_equation(Y)

    def stamp_dc(self, Y, sources):
        """In DC, an inductor is a short circuit (V1 - V2 = 0)."""
        # The _stamp_branch_equation already handled the Y matrix.
        # We just ensure the RHS is 0.
        sources[self.branch_idx] = 0.0

    def stamp_ac(self, Y, sources, w):
        """Stamps complex impedance: Z = j * w * L."""
        z = 1j * w * self.value
        if self.branch_idx is not None:
            Y[self.branch_idx, self.branch_idx] -= z

    def stamp_transient(self, Y, sources, t, dt, v_prev):
        """Backward Euler Companion Model: V(t) = (L/dt)*(I(t) - I(prev))."""
        req = self.value / dt
        # Current is stored in the solution vector at the branch_idx
        i_prev = v_prev[self.branch_idx] if self.branch_idx is not None else 0.0
        v_eq = req * i_prev
        
        if self.branch_idx is not None:
            Y[self.branch_idx, self.branch_idx] -= req
            sources[self.branch_idx] -= v_eq

    def build_adjoint_history(self, J_hist, dt, v_hat_next, adjoint_state, method='BE'):
        """Builds the adjoint RHS memory term for Backward Euler."""
        if self.branch_idx is not None:
            # The adjoint inductor memory depends on the adjoint branch current 
            # from the 'next' time step (which we already solved in backward time)
            i_L_hat = v_hat_next[self.branch_idx]
            
            # Adjoint Equivalent Voltage Source: V_eq_hat = (L/dt) * I_L_hat
            V_eq_hat = (self.value / dt) * i_L_hat
            
            # Subtract from the branch equation row in the RHS history
            J_hist[self.branch_idx] -= V_eq_hat

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivity w.r.t Inductance (L)."""
        if self.branch_idx is None: return {}

        # The current flowing through the inductor is stored at the branch index
        i_L = VI[self.branch_idx]
        
        # Adjoint branch variable
        psi_branch = PsiPhi[self.branch_idx]

        # 1. AC Analysis (V_L = j * w * L * I_L)
        if w != 0.0:
            dVL_dL = 1j * w * i_L
            return {self.name: psi_branch * dVL_dL}

        # 2. Transient Analysis (V_L = (L/dt) * (I_L - I_prev))
        elif dt is not None and V_prev is not None:
            i_L_prev = V_prev[self.branch_idx]
            
            # dVL/dL = dI/dt using Backward Euler
            dI_dt = (i_L - i_L_prev) / dt
            return {self.name: psi_branch * dI_dt}

        # 3. DC Analysis (Inductor is a short, L has no effect on DC bias)
        else:
            return {self.name: 0.0}

# =====================================================================
# NONLINEAR COMPONENTS
# =====================================================================
class Diode(Component):
    """Nonlinear Diode (Type 'D') using hard-coded MNA stamping."""

    def bind_nodes(self, node_map):
        """Maps Anode (n1) and Cathode (n2) to matrix indices."""
        self.idx_a = node_map.get(self.data.get("n1", 0))
        self.idx_k = node_map.get(self.data.get("n2", 0))
        
        params = self.data.get("model_params", {})
        self.IS = params.get("IS", 1e-14)
        self.VT = params.get("VT", 0.02585)

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        """Stamps linearized gd and Ieq into the MNA system."""
        # 1. Extract voltages
        va = V_guess[self.idx_a] if self.idx_a is not None else 0.0
        vk = V_guess[self.idx_k] if self.idx_k is not None else 0.0
        vd = va - vk

        # 2. Evaluate Physics from models.py
        res = models.evaluate_diode(vd, self.IS, self.VT)
        id_val, gd = res["I_D"], res["gd"]

        # 3. Newton-Raphson linearized current source
        # Ieq = Id - gd * Vd
        ieq = id_val - gd * vd

        # 4. Hard-Coded "Slot" Stamping
        # --- Anode Row ---
        if self.idx_a is not None:
            Y[self.idx_a, self.idx_a] += gd
            sources[self.idx_a] -= ieq
            if self.idx_k is not None:
                Y[self.idx_a, self.idx_k] -= gd

        # --- Cathode Row ---
        if self.idx_k is not None:
            Y[self.idx_k, self.idx_k] += gd
            sources[self.idx_k] += ieq
            if self.idx_a is not None:
                Y[self.idx_k, self.idx_a] -= gd

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivity w.r.t Saturation Current (IS)."""
        va = VI[self.idx_a] if self.idx_a is not None else 0.0
        vk = VI[self.idx_k] if self.idx_k is not None else 0.0
        
        pa = PsiPhi[self.idx_a] if self.idx_a is not None else 0.0
        pk = PsiPhi[self.idx_k] if self.idx_k is not None else 0.0
        
        res = models.evaluate_diode(va - vk, self.IS, self.VT)
        
        # d(Output)/d(IS) = (Psi_anode - Psi_cathode) * d(Id)/d(IS)
        return {f"{self.name}_IS": (pa - pk) * res["dId_dIs"]}


class Mosfet(Component):
    """Simple Level 1 NMOS model with Hard-Coded Stamping."""

    def bind_nodes(self, node_map):
        self.idx_d = node_map.get(self.data.get("n_d", 0))
        self.idx_g = node_map.get(self.data.get("n_g", 0))
        self.idx_s = node_map.get(self.data.get("n_s", 0))
        
        params = self.data.get("model_params", {})
        inst = self.data.get("inst_params", {})
        
        self.VTO = params.get("VTO", 0.7)
        self.W = inst.get("W", 1e-6)
        self.L = inst.get("L", 1e-6)
        
        # Process parameters
        mu = params.get("MU", 600e-4) # default mobility
        cox = params.get("C_OX", 3.45e-3) # default oxide capacitance
        self.KP = params.get("KP", mu * cox)
        
        # Bn = (W/L) * KP
        self.Bn = (self.W / self.L) * self.KP

    def stamp_nonlinear(self, Y, sources, p_V_guess, V_guess):
        # 1. Get current iteration voltages
        vd = V_guess[self.idx_d] if self.idx_d is not None else 0.0
        vg = V_guess[self.idx_g] if self.idx_g is not None else 0.0
        vs = V_guess[self.idx_s] if self.idx_s is not None else 0.0
        
        # 2. Evaluate Physics
        res = models.evaluate_nmos(vg - vs, vd - vs, self.VTO, self.Bn)
        Id, gm, gds = res["I_D"], res["gm"], res["gds"]

        # 3. Hard-Coded Stamping (The "Slot" Method)
        # We define Ieq to handle the Newton-Raphson linearization offset
        ieq = Id - gm * (vg - vs) - gds * (vd - vs)

        # Drain terminal slots
        if self.idx_d is not None:
            if self.idx_g is not None: Y[self.idx_d, self.idx_g] += gm
            if self.idx_s is not None: Y[self.idx_d, self.idx_s] -= (gm + gds)
            Y[self.idx_d, self.idx_d] += gds
            sources[self.idx_d] -= ieq

        # Source terminal slots
        if self.idx_s is not None:
            if self.idx_g is not None: Y[self.idx_s, self.idx_g] -= gm
            if self.idx_d is not None: Y[self.idx_s, self.idx_d] -= gds
            Y[self.idx_s, self.idx_s] += (gm + gds)
            sources[self.idx_s] += ieq

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivities for W, L, and VTO using the Adjoint method."""
        vd = VI[self.idx_d] if self.idx_d is not None else 0.0
        vg = VI[self.idx_g] if self.idx_g is not None else 0.0
        vs = VI[self.idx_s] if self.idx_s is not None else 0.0
        
        pd = PsiPhi[self.idx_d] if self.idx_d is not None else 0.0
        ps = PsiPhi[self.idx_s] if self.idx_s is not None else 0.0
        
        # Adjoint term: potential difference across the 'current' of the device
        adj_factor = (pd - ps)
        
        res = models.evaluate_nmos(vg - vs, vd - vs, self.VTO, self.Bn)
        dId_dBn = res["dId_dBn"]
        dId_dVTO = res["dId_dVTO"]

        # Apply Chain Rule
        sens_W = adj_factor * dId_dBn * (self.KP / self.L)
        sens_L = adj_factor * dId_dBn * (-self.W * self.KP / (self.L**2))
        sens_VTO = adj_factor * dId_dVTO
        
        return {
            f"{self.name}_W": sens_W,
            f"{self.name}_L": sens_L,
            f"{self.name}_VTO": sens_VTO
        }

class OpAmp(Component):
    """
    Ideal Op-Amp (Type 'E'). 
    Implemented as a Voltage-Controlled Voltage Source (VCVS).
    V(out, gnd) = Gain * (V(n_plus) - V(n_minus))
    """

    def bind_nodes(self, node_map):
        # Input terminals
        self.idx_p = node_map.get(self.data.get("n1", 0)) # Non-inverting (+)
        self.idx_m = node_map.get(self.data.get("n2", 0)) # Inverting (-)
        
        # Output terminal
        self.idx_out = node_map.get(self.data.get("n_out", 0))
        
        # Branch index for the VCVS equation
        self.branch_idx = node_map.get(self.name)
        
        # High open-loop gain (default 100k if not specified)
        self.gain = self.data.get("value", 1e5)

    def stamp_mna_connection(self, Y):
        """
        Stamps the VCVS MNA equations.
        Equation: V_out - Gain*(V_p - V_m) = 0
        """
        if self.branch_idx is None: return

        # 1. Output current flows into the output node
        if self.idx_out is not None:
            Y[self.idx_out, self.branch_idx] += 1.0
            Y[self.branch_idx, self.idx_out] += 1.0

        # 2. Control voltage dependencies in the branch row
        if self.idx_p is not None:
            Y[self.branch_idx, self.idx_p] -= self.gain
        if self.idx_m is not None:
            Y[self.branch_idx, self.idx_m] += self.gain

    # def stamp_static(self, Y, sources):
    #     # RHS for an ideal Op-Amp is typically 0 (homogeneous equation)
    #     if self.branch_idx is not None:
    #         sources[self.branch_idx] = 0.0

    # # AC and Transient inherit from static
    # def stamp_ac(self, Y, sources, w): self.stamp_static(Y, sources)
    # def stamp_transient(self, Y, sources, t, dt, v_prev): self.stamp_static(Y, sources)

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None):
        """Calculates sensitivity w.r.t Open-Loop Gain (A)."""
        if self.branch_idx is None: return {}

        # Forward differential input
        vp = VI[self.idx_p] if self.idx_p is not None else 0.0
        vm = VI[self.idx_m] if self.idx_m is not None else 0.0
        v_diff = vp - vm

        # Adjoint branch variable
        psi_branch = PsiPhi[self.branch_idx]

        # Adjoint formula: Psi_branch * (V_plus - V_minus)
        return {self.name: psi_branch * v_diff}
