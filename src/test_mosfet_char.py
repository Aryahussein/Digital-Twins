import numpy as np
import matplotlib.pyplot as plt

# --- 1. ROBUST FUNCTIONS (From previous fixes) ---

def phi_f(x):
    """ Robust interpolation function [ln(1+e^(x/2))]^2 """
    limit = 50.0
    # Clamp x to avoid overflow in exp()
    x_safe = np.minimum(x, limit)
    
    # Calculate both possibilities
    small_val = np.log(1 + np.exp(x_safe / 2.0))**2
    large_val = (x / 2.0)**2
    
    # Select based on threshold
    return np.where(x > limit, large_val, small_val)

def mos_textbook(vgs, vds):
    """ Standard Square-Law Model (Broadcast Safe) """
    # Ensure inputs are arrays of same shape
    vgs = np.asarray(vgs)
    vds = np.asarray(vds)
    
    # If they are different shapes (e.g. scalar vs array), broadcast them
    if vgs.shape != vds.shape:
        vgs, vds = np.broadcast_arrays(vgs, vds)
    
    Id = np.zeros_like(vgs)
    gm = np.zeros_like(vgs)
    
    # Physics Parameters
    W, L = 10e-6, 1e-6
    Kp = 120e-6
    Beta = (W/L) * Kp
    Vth = 0.5
    
    vov = vgs - Vth
    
    # Masks
    is_active = (vov > 0)
    is_linear = is_active & (vds < vov)
    is_sat = is_active & (vds >= vov)
    
    # Linear
    if np.any(is_linear):
        vds_l = vds[is_linear]
        vov_l = vov[is_linear]
        Id[is_linear] = Beta * (vov_l * vds_l - 0.5 * vds_l**2)
        gm[is_linear] = Beta * vds_l 

    # Saturation
    if np.any(is_sat):
        vov_s = vov[is_sat]
        Id[is_sat] = 0.5 * Beta * (vov_s**2)
        gm[is_sat] = Beta * vov_s
            
    return Id, gm

def mos_ekv(vgs, vds):
    """ Analytic Differentiable Model """
    vgs = np.asarray(vgs)
    vds = np.asarray(vds)
    if vgs.shape != vds.shape:
        vgs, vds = np.broadcast_arrays(vgs, vds)
        
    # Physics Parameters
    W, L = 10e-6, 1e-6
    Kp = 120e-6
    Ut = 0.02585
    n = 1.0
    Beta = (W/L) * Kp
    Is = 2 * n * Beta * (Ut**2)
    Vth = 0.5
    
    # Normalized Voltages
    vp = (vgs - Vth) / (n * Ut)
    vs = 0.0
    vd = vds / Ut
    
    # Currents
    if_val = phi_f(vp - vs)
    ir_val = phi_f(vp - vd)
    Id = Is * (if_val - ir_val)
    
    # GM Calculation (Numerical for simplicity)
    eps = 1e-5
    vp_eps = ((vgs + eps) - Vth) / (n * Ut)
    Id_plus = Is * (phi_f(vp_eps - vs) - phi_f(vp_eps - vd))
    gm = (Id_plus - Id) / eps
    
    return Id, gm

# --- PLOTTING ---

# Common Data
vgs_sweep = np.linspace(0, 1.2, 500)
vds_sat = np.full_like(vgs_sweep, 1.0) # Saturation
vds_lin = np.full_like(vgs_sweep, 0.05) # Linear

# ==========================================
# PLOT 1: Transfer (Id vs Vgs) - Dual Axis
# ==========================================
fig1, ax1 = plt.subplots(figsize=(4, 3))

# Calculate Data
id_txt, _ = mos_textbook(vgs_sweep, vds_sat)
id_ekv, _ = mos_ekv(vgs_sweep, vds_sat)

# Left Axis: Linear Scale ax1.plot(vgs_sweep, id_ekv * 1e6, 'k-', label='Analytic (Linear)') ax1.plot(vgs_sweep, id_txt * 1e6, 'k-',label='Piecewise (Linear)')
ax1.plot(vgs_sweep, id_ekv * 1e6, 'k-', label='Analytic (Linear)')
ax1.set_ylabel('Id [uA] (Linear)')
ax1.grid(True, alpha=0.3)

# Right Axis: Log Scale
ax2 = ax1.twinx()
# Add epsilon to log plot to avoid log(0)
ax2.semilogy(vgs_sweep, id_ekv, 'k--', label='Analytic (Log)', linewidth=1.5)
ax2.semilogy(vgs_sweep, id_txt + 1e-12, 'r-', label='Piecewise (Log)')
ax2.set_ylabel('Id [A] (Log)')

# Combined Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
fig1.savefig('../figures/mosfet_char/test_mosfet_char.png', dpi=600, bbox_inches='tight')


# ==========================================
# PLOT 2: Output (Id vs Vds) - Step Vgs
# ==========================================
fig2, ax3 = plt.subplots(figsize=(4, 3))

vds_sweep = np.linspace(0, 1.2, 200)
vgs_steps = [0.6, 0.8, 1.0]

for vgs in vgs_steps:
    # Create array of same size
    vgs_arr = np.full_like(vds_sweep, vgs)
    
    id_t, _ = mos_textbook(vgs_arr, vds_sweep)
    id_e, _ = mos_ekv(vgs_arr, vds_sweep)
    
    # ax3.plot(vds_sweep, id_e * 1e6, 'k-', label=f'Vgs={vgs}V') # EKV
    ax3.plot(vds_sweep, id_t * 1e6, label=f'Vgs={vgs}V') # Textbook

ax3.set_xlabel('Vds [V]')
ax3.set_ylabel('Id [uA]')
ax3.grid(True, alpha=0.3)
# Custom legend to explain colors
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='r', lw=2, linestyle='--')]
# ax3.legend(custom_lines, ['Analytic', 'Piecewise'])
ax3.legend(loc='upper right')
fig2.savefig('../figures/mosfet_char/test_mosfet_char_ID_Vds.png', dpi=600, bbox_inches='tight')

# ==========================================
# PLOT 3: Efficiency (gm/Id vs Vgs)
# ==========================================
fig3, ax4 = plt.subplots(figsize=(4, 3))

# Use log-spaced Vgs for better view of weak inversion
vgs_eff = np.linspace(0.1, 1.0, 500)
vds_eff = np.full_like(vgs_eff, 1.0)

id_t, gm_t = mos_textbook(vgs_eff, vds_eff)
id_e, gm_e = mos_ekv(vgs_eff, vds_eff)

# Avoid division by zero
eff_t = np.divide(gm_t, id_t, out=np.zeros_like(gm_t), where=id_t>1e-15)
eff_e = np.divide(gm_e, id_e, out=np.zeros_like(gm_e), where=id_e>1e-15)

ax4.plot(vgs_eff, eff_e, 'k-', label='Analytic')
ax4.plot(vgs_eff, eff_t, 'r--', label='Piecewise')

ax4.set_xlabel('Vgs [V]')
ax4.set_ylabel('gm / Id [S/A]')
ax4.set_ylim(0, 45) # Theoretical limit is ~38.6
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.show()
