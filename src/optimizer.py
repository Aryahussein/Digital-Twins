import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_linear_design_space(V_current, V_target, current_params, sensitivities):
    """
    Calculates the required change in output and isolates the linear plane.
    current_params: dict of {'M1_W': 1e-6, 'M2_W': 2e-6, 'M3_W': 1.5e-6}
    sensitivities: dict of the Integrated Adjoint scalars for those 3 parameters
    """
    delta_V = V_target - V_current
    print(f"Targeting a voltage shift of: {delta_V:+.4f} V")
    return delta_V

def plot_3d_design_space(current_params, sensitivities, delta_V, param_bounds=0.2):
    """
    Plots the 2D design plane inside a 3D parameter space.
    param_bounds: +/- 20% variation on the X and Y axes for the plot
    """
    # 1. Extract the names and values
    names = list(current_params.keys())
    if len(names) != 3:
        raise ValueError("This visualization requires exactly 3 parameters!")
        
    p1_name, p2_name, p3_name = names
    p1_0, p2_0, p3_0 = current_params[p1_name], current_params[p2_name], current_params[p3_name]
    s1, s2, s3 = sensitivities[p1_name], sensitivities[p2_name], sensitivities[p3_name]

    if s3 == 0:
        raise ValueError(f"Sensitivity of {p3_name} is 0. Cannot divide by zero to plot plane.")

    # 2. Create the grid for Parameter 1 and Parameter 2 (+/- 20% of their current values)
    p1_range = np.linspace(p1_0 * (1 - param_bounds), p1_0 * (1 + param_bounds), 30)
    p2_range = np.linspace(p2_0 * (1 - param_bounds), p2_0 * (1 + param_bounds), 30)
    P1, P2 = np.meshgrid(p1_range, p2_range)

    # 3. Calculate Parameter 3 using the Taylor Series Hyperplane Equation:
    # delta_V = S1*(P1 - p1_0) + S2*(P2 - p2_0) + S3*(P3 - p3_0)
    # Solving for P3:
    P3 = p3_0 + (delta_V - s1*(P1 - p1_0) - s2*(P2 - p2_0)) / s3

    # 4. Plotting the 3D Plane
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the design space plane
    surf = ax.plot_surface(P1, P2, P3, cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Plot the CURRENT state as a red dot
    ax.scatter(p1_0, p2_0, p3_0, color='red', s=100, label='Current Design (Failing)')
    
    # Labels and formatting
    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_zlabel(p3_name)
    ax.set_title(f"Feasible Design Space to shift output by {delta_V:+.3f}V")
    
    # Format axes to scientific notation for tiny component values
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend()
    plt.show()

# =====================================================================
# SKELETON INTEGRATION: How you call this from your main file
# =====================================================================
def run_optimization_tool():
    # 1. You would run your simulator here to get the current state
    # results = run_simulation_core(...)
    
    # Let's mock the data your simulator just outputted:
    V_current = 2.0  # The voltage your simulator actually hit at the final timestep
    V_target  = 2.5  # The voltage you WANTED it to hit
    
    # The current widths of the 3 transistors you want to tweak (e.g., in meters)
    current_params = {
        'M1_W': 1.0e-6,
        'M2_W': 2.0e-6,
        'M3_W': 1.5e-6
    }
    
    # The "Integrated_Transient" Adjoint Sensitivities your code just calculated
    sensitivities = {
        'M1_W': -50000.0, # Increasing M1_W drops the voltage heavily
        'M2_W': 25000.0,  # Increasing M2_W raises the voltage moderately
        'M3_W': 10000.0   # Increasing M3_W raises the voltage slightly
    }
    
    # 2. Calculate the required shift
    delta_V = get_linear_design_space(V_current, V_target, current_params, sensitivities)
    
    # 3. Plot the valid design space!
    plot_3d_design_space(current_params, sensitivities, delta_V, param_bounds=0.5)

if __name__ == "__main__":
    run_optimization_tool()
