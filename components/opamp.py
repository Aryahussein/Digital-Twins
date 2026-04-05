"""Op-Amp (VCVS) Component Module.

Implements a Voltage-Controlled Voltage Source:
    V(out+) - V(out-) = Gain * (V(ctrl+) - V(ctrl-))
using a standard MNA branch equation formulation.

Standard SPICE format: E<n> N_OUT+ N_OUT- NC+ NC- GAIN

The MNA branch equation introduces one additional unknown (the branch
current flowing from out+ to out-) and enforces the voltage relationship.
This handles both grounded outputs (N_OUT- = 0, typical opamp) and 
floating outputs (N_OUT- ≠ 0, general VCVS).
"""

from .base import Component


class OpAmp(Component):
    """Ideal Op-Amp / VCVS (Type 'E').
    
    This is a LINEAR element — does not require NR iteration.
    
    MNA stamp structure for E1 out+ out- ctrl+ ctrl- A:
        Branch equation row:  V(out+) - V(out-) - A*V(ctrl+) + A*V(ctrl-) = 0
        KCL at out+:         +I_branch
        KCL at out-:         -I_branch
    
    Matrix entries:
        Y[out+, branch] = +1    Y[branch, out+]  = +1
        Y[out-, branch] = -1    Y[branch, out-]  = -1
        Y[branch, ctrl+] = -A
        Y[branch, ctrl-] = +A
    """

    IS_NONLINEAR = False
    NODE_KEYS = ("n1", "n2", "n_out", "n_out_m")

    def bind_nodes(self, node_map):
        # Control terminals
        self.idx_p = node_map.get(self.data.get("n1", 0))      # Control + (non-inverting)
        self.idx_m = node_map.get(self.data.get("n2", 0))      # Control - (inverting)
        
        # Output terminals
        self.idx_out = node_map.get(self.data.get("n_out", 0))    # Output +
        self.idx_out_m = node_map.get(self.data.get("n_out_m", 0))  # Output -
        
        # Branch index for the VCVS equation
        self.branch_idx = node_map.get(self.name)
        
        # Open-loop gain (default 100k if not specified)
        self.gain = self.data.get("value", 1e5)

    def stamp_mna_connection(self, Y):
        """Stamps the VCVS MNA equations.
        
        Branch equation: V(out+) - V(out-) - A*(V(ctrl+) - V(ctrl-)) = 0
        Branch current flows from out+ through the source to out-.
        """
        if self.branch_idx is None: 
            return

        b = self.branch_idx

        # Output positive terminal: current enters out+
        if self.idx_out is not None:
            Y[self.idx_out, b] += 1.0
            Y[b, self.idx_out] += 1.0

        # Output negative terminal: current leaves out-
        if self.idx_out_m is not None:
            Y[self.idx_out_m, b] -= 1.0
            Y[b, self.idx_out_m] -= 1.0

        # Control voltage dependencies in the branch row
        if self.idx_p is not None:
            Y[b, self.idx_p] -= self.gain
        if self.idx_m is not None:
            Y[b, self.idx_m] += self.gain

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity w.r.t. Open-Loop Gain (A).
        
        Branch equation: V(out+) - V(out-) = A * (V(ctrl+) - V(ctrl-))
        Derivative w.r.t. A: d/dA = -(V(ctrl+) - V(ctrl-)) in the branch row
        Adjoint formula: sens = -Psi_branch * (-(V(ctrl+) - V(ctrl-)))
                              = Psi_branch * (V(ctrl+) - V(ctrl-))
        """
        if self.branch_idx is None: 
            return {}

        vp = VI[self.idx_p] if self.idx_p is not None else 0.0
        vm = VI[self.idx_m] if self.idx_m is not None else 0.0
        v_diff = vp - vm

        psi_branch = PsiPhi[self.branch_idx]

        return {self.name: psi_branch * v_diff}
