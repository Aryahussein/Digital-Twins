"""
AWE Engine — Asymptotic Waveform Evaluation
============================================
Builds a Padé approximant from circuit moments and evaluates the
resulting rational transfer function at any frequency or time point.

For a circuit with transfer function H(s) = V_out(s)/V_in(s), AWE
constructs a rational approximation matching the first 2q moments:

    H(s) ≈ (a_0 + a_1*s + ... + a_{q-1}*s^{q-1})
           ─────────────────────────────────────────
           (1   + b_1*s + ... + b_q*s^q)

Denominator coefficients b_1..b_q from the linear system:

    sum_{j=1}^{q}  b_j * m_{k-j} = -m_k    for k = q, q+1, ..., 2q-1

Numerator coefficients a_k:

    a_k = m_k + sum_{j=1}^{k} b_j * m_{k-j}    for k = 0, ..., q-1

Stability check: all poles must have negative real part.
If not, automatically reduces order by 1 until stable.
"""

import numpy as np
from engines.moments_engine import MomentsEngine


class AWEEngine:
    """AWE: Padé approximant from circuit moments."""

    def __init__(self, circuit, output_node, order=4):
        self.circuit     = circuit
        self.output_node = output_node
        self.order       = order

        self.poles    = None
        self.residues = None
        self.a_coeffs = None
        self.b_coeffs = None
        self.moments  = None

    def _build_pade(self, order=None):
        """
        Compute moments and build the Padé approximant.
        Reduces order by 1 if unstable poles detected.
        Returns True when done.
        """
        if order is None:
            order = self.order

        q           = order
        num_moments = 2 * q

        # Compute moments m_0 ... m_{2q-1}
        mom_eng      = MomentsEngine(self.circuit)
        moments, _   = mom_eng.compute(self.output_node, num_moments)
        self.moments = moments

        # ── Denominator system ──────────────────────────────────────
        # For k = q, q+1, ..., 2q-1:
        #   sum_{j=1}^{q} b_j * m_{k-j} = -m_k
        # Matrix: M[i, j-1] = m_{(q+i) - j}   (i=0..q-1, j=1..q)
        # RHS:    rhs[i]     = -m_{q+i}

        M   = np.zeros((q, q))
        rhs = np.zeros(q)

        for i in range(q):
            rhs[i] = -moments[q + i]
            for j in range(1, q + 1):
                idx = (q + i) - j          # = q+i-j, always >= 0
                M[i, j - 1] = moments[idx] if 0 <= idx < num_moments else 0.0

        try:
            b = np.linalg.solve(M, rhs)    # b[j-1] = b_j
        except np.linalg.LinAlgError:
            # Singular Padé matrix — use least-squares solution
            b, _, rank, _ = np.linalg.lstsq(M, rhs, rcond=None)
            if rank < q:
                # System is genuinely rank-deficient, reduce order
                if q > 1:
                    return self._build_pade(order=q - 1)
                b = np.zeros(q)

        # ── Numerator coefficients ──────────────────────────────────
        # a_k = m_k + sum_{j=1}^{k} b_j * m_{k-j}   for k=0..q-1
        a = np.zeros(q)
        for k in range(q):
            a[k] = moments[k]
            for j in range(1, k + 1):
                a[k] += b[j - 1] * moments[k - j]

        # ── Denominator polynomial and poles ────────────────────────
        # Q(s) = 1 + b_1*s + ... + b_q*s^q
        # numpy.roots expects coefficients from HIGHEST to LOWEST degree:
        # Q_np = [b_q, b_{q-1}, ..., b_1, 1]
        Q_np = np.zeros(q + 1)
        Q_np[-1] = 1.0                     # constant term
        for j in range(1, q + 1):
            Q_np[q - j] = b[j - 1]        # b_j is coeff of s^j

        poles = np.roots(Q_np)

        # Handle q=0 edge case (all orders failed — use DC only)
        if len(poles) == 0:
            self.poles    = np.array([])
            self.residues = np.array([])
            self.a_coeffs = a
            self.b_coeffs = b
            return True

        # ── Stability check ─────────────────────────────────────────
        # Poles with small positive real part may be numerical noise from
        # nearly-singular Padé systems. Use a relative threshold.
        max_real = np.max(np.real(poles))
        min_real = np.min(np.real(poles))
        # Unstable if any pole clearly in RHP (not just numerical noise)
        unstable = max_real > 1e-6 * abs(min_real) and max_real > 0

        if unstable:
            if q > 1:
                print(f"  [AWE] Unstable at order {q} "
                      f"({np.sum(np.real(poles) > 0)} RHP poles). "
                      f"Reducing to order {q-1}.")
                return self._build_pade(order=q - 1)
            else:
                print(f"  [AWE] Warning: order-1 still unstable "
                      f"(pole={poles[0]:.3e}). Using anyway.")

        # ── Residues via partial fractions ──────────────────────────
        # P(s) = a_0 + a_1*s + ... + a_{q-1}*s^{q-1}
        # numpy poly: [a_{q-1}, ..., a_1, a_0]
        P_np = a[::-1].copy()

        # Q'(s) = derivative of Q
        Q_deriv = np.polyder(Q_np)

        residues = np.zeros(len(poles), dtype=complex)
        for i, p in enumerate(poles):
            P_at_p  = np.polyval(P_np,  p)
            Qd_at_p = np.polyval(Q_deriv, p)
            residues[i] = P_at_p / Qd_at_p if abs(Qd_at_p) > 1e-30 else 0.0

        self.poles    = poles
        self.residues = residues
        self.a_coeffs = a
        self.b_coeffs = b
        return True

    def build(self):
        """Build the Padé approximant. Call before evaluate_*."""
        self._build_pade()
        print(f"  [AWE] Built order-{len(self.poles)} Padé approximant")
        print(f"  [AWE] Poles: {np.round(self.poles, 4)}")
        return self

    def evaluate_freq(self, frequencies):
        """
        Evaluate H(jω) at given frequencies using partial-fraction form:

            H(s) = sum_i  residue_i / (s - pole_i)
        """
        if self.poles is None:
            self.build()

        omegas = 2 * np.pi * np.asarray(frequencies, dtype=float)
        H = np.zeros(len(omegas), dtype=complex)
        for p, r in zip(self.poles, self.residues):
            H += r / (1j * omegas - p)
        return H

    def evaluate_time(self, time_array):
        """
        Step response via inverse Laplace of H(s)/s.

        H(s)/s = sum_i  r_i / (s * (s - p_i))
               = sum_i  [(-r_i/p_i)/s  +  (r_i/p_i)/(s - p_i)]

        Inverse Laplace:
            v(t) = sum_i  (r_i/p_i) * (exp(p_i * t) - 1)

        This evaluates to 0 at t=0 and approaches m_0 as t → ∞,
        which is correct for a step response starting from 0.
        """
        if self.poles is None:
            self.build()

        t = np.asarray(time_array, dtype=float)
        v = np.zeros(len(t), dtype=complex)
        for p, r in zip(self.poles, self.residues):
            if abs(p) > 1e-30:
                v += (r / p) * (np.exp(p * t) - 1)
        return np.real(v)

    def report(self):
        """Print a formatted summary of poles, residues, and time constants."""
        if self.poles is None:
            self.build()
        print("\n=== AWE Report ===")
        print(f"  Order: {len(self.poles)}")
        print(f"  {'Pole':<25} {'Residue':<25} {'Time const (s)'}")
        for p, r in zip(self.poles, self.residues):
            tau = -1.0/np.real(p) if np.real(p) < 0 else float('inf')
            print(f"  {str(np.round(p,4)):<25} "
                  f"{str(np.round(r,4)):<25} "
                  f"{tau:.4e}")
