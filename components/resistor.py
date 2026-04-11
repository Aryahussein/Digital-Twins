"""Resistor Component Module."""

from .base import Component


class Resistor(Component):
    """A linear resistor component (Type 'R').
    
    MNA stamp: Conductance G = 1/R is added as a 2x2 block to the 
    admittance matrix at the (n1, n2) node pair.
    """

    def _validate(self):
        if self.value == 0.0:
            raise ValueError(
                f"Resistor '{self.name}' has value 0 Ohms. "
                "This creates infinite conductance (singular matrix). "
                "Use a very small value (e.g., 1e-9) instead."
            )
        if self.value < 0.0:
            raise ValueError(
                f"Resistor '{self.name}' has negative resistance ({self.value}). "
                "Negative resistors are not supported."
            )

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

    def get_sensitivities(self, VI, PsiPhi, w=0.0, dt=None, V_prev=None, method='BE'):
        """Sensitivity of output w.r.t. resistance R.
        
        Derivation:
            The resistor stamps G = 1/R into the MNA matrix.
            dG/dR = -1/R^2
            Adjoint formula: sens = -(Psi1 - Psi2) * (V1 - V2) * dG/dR
                                   = -(Psi1 - Psi2) * (V1 - V2) * (-1/R^2)
                                   = (Psi1 - Psi2) * (V1 - V2) / R^2
            
            The double negative (from dG/dR and the adjoint sign) cancels,
            yielding a positive product — no explicit minus sign needed.
        """
        v1 = VI[self.idx_1] if self.idx_1 is not None else 0.0
        v2 = VI[self.idx_2] if self.idx_2 is not None else 0.0
        p1 = PsiPhi[self.idx_1] if self.idx_1 is not None else 0.0
        p2 = PsiPhi[self.idx_2] if self.idx_2 is not None else 0.0
        
        return {self.name: (1.0 / (self.value**2)) * ((v1 - v2) * (p1 - p2))}

    def get_noise_sources(self, VI, w):
        """Thermal (Johnson-Nyquist) noise: S_i = 4kT/R  A²/Hz."""
        from core.constants import kb, T
        S = 4.0 * kb * T / self.value
        return [{'nodes': (self.idx_1, self.idx_2),
                 'S': S,
                 'label': f'{self.name}_thermal'}]
