from txt2dictionary import parse_netlist
from node_index import build_node_index
from solver import solve_sparse, solve_LU, solve_adjoint, get_node_and_branch_currents
from postprocessing import map_voltages
from assembleYmatrix import generate_stamps
import numpy as np
from tools import run_bode_plot, print_solution, get_all_sensitivities, plot_sensitivity_sweep, plot_transient
from transientsolver import stamp_capacitor_transient

def build_matrix(componenents, w=0):
    node_map, total_dim = build_node_index(components)

    # Build Y matrix and I vector from components
    Y, sources = generate_stamps(components, node_map, total_dim, w=w)

    return Y, sources, node_map, total_dim

def estimate_std_dev(sensitivities, components, percent_sigma=0.01):
    variance = 0
    for name, sens in sensitivities.items():
        # Assume 1% is the standard deviation of the component
        sigma_p = components[name]["value"] * percent_sigma
        variance += np.abs(sens * sigma_p)**2
        
    return np.sqrt(variance)

def do_sensitivity_analysis(lu, VI, output_node, node_map, total_dim, w=0):
    # get all node and branch voltages of adjoint circuit
    PsiPhi = solve_adjoint(lu, output_node, node_map, total_dim)

    # get the sensitivities of all components
    sensitivities = get_all_sensitivities(components, VI, PsiPhi, node_map, w=w)
    # print(sensitivities)

    # find the standard deviation on the output voltage
    tolerance_on_components = 0.01
    std_dev = estimate_std_dev(sensitivities, components, percent_sigma=tolerance_on_components)
    # print(std_dev)
    return sensitivities, std_dev






if __name__ == "__main__":
    test_directory = "testfiles/"
    netlist = test_directory + "/rc.txt"

    ############ w = 2*np.pi * 60
    w = 0

    # get components
    components = parse_netlist(netlist)
    print(components)

    # get matrix, sources, and mapping of components to matrix indeces
    Y, sources, node_map, total_dim = build_matrix(components, w=w)
    print(node_map)

    # solve! get LU, node voltages and branch currents of original circuit
    lu = solve_LU(Y)
    VI = get_node_and_branch_currents(lu, sources)
    ###print_solution(VI, node_map, w=w)




    ############ Transient solving ############

    # transient-time variables
    num_steps = 100000
    transient_timestep = 1/num_steps
    transient_time = 0
    store_voltage = np.zeros(1000000)
    store_current = np.zeros(1000000)
    store_time = np.zeros(1000000)

    ########### use a fake solve where the current is 0A and voltage is 0A to mimic an initial condition where Vsource=10V but capacitor is charged to 1V
    VI = np.zeros(total_dim)
    VI[0] = 10
    VI[1] = 1
    #print(VI)
    print("/n/n/n")
    print_solution(VI, node_map, w=w)

    # DC-solved "Y" and "sources" matrices are permanent, but we add the Norton equivalent for Back Euler Approximation to it 
        # Norton equivalent is used
    Y_temp = Y.copy()
    sources_temp = sources.copy()

    
    store_time[1] = transient_timestep

    # Step through transient time 'num_steps' number of times
    for tran_step_count in range(1, num_steps):
        ###print(f"\n\n\n tran_step_count = {tran_step_count}")
        store_time[tran_step_count] = transient_timestep + store_time[tran_step_count-1]
        
        # Reset the Y_temp matrix to Y matrix (which then has the capacitor/inductor stamps added onto it)
        Y_temp = Y.copy()

        # Reset the sources_temp matrix to sources matrix (which then has the capacitor/inductor stamps added onto it)
        sources_temp = sources.copy()

        # Stamp capacitor & inductor Back Euler approximations
        for name, comp in components.items():
            val = comp["value"]

            n1 = comp.get("n1", 0)
            n2 = comp.get("n2", 0)

            # Stamp the Capacitor Back Euler approximation
            if name.startswith("C"):
                
                # Find the voltage from the previous timestep solved
                if n1:
                    cap_voltage_n1 = VI[n1-1]
                else:                               # voltage is 0V if n2 is ground
                    cap_voltage_n1 = 0
                if n2:
                    cap_voltage_n2 = VI[n2-1]
                else:                               # voltage is 0V if n2 is ground
                    cap_voltage_n2 = 0

                cap_voltage = cap_voltage_n2 - cap_voltage_n1

                ###print(f"cap_voltage = {cap_voltage} where \n       n1_index is {n1} and cap_voltage_n1= {cap_voltage_n1}  \n       n2_index is {n2} and cap_voltage_n2= {cap_voltage_n2}")

                # Stamp capacitor admittance and add current for Back Euler approximation
                stamp_capacitor_transient(Y_temp, sources_temp, n1, n2, val, node_map, transient_timestep, cap_voltage)

        lu = solve_LU(Y_temp)
        VI = get_node_and_branch_currents(lu, sources_temp)

        ###print_solution(VI, node_map, w=w)

        store_voltage[tran_step_count] = VI[1]
        store_current[tran_step_count] = VI[2]


    
    
    
    
    ###lu = solve_LU(Y_temp)
    ###VI_temp = get_node_and_branch_currents(lu, sources_temp)
    ###print_solution(VI_temp, node_map, w=w)
    
    for i in range(0, num_steps):
        if i % 1000 == 0:
            print(f"Time = {store_time[i]:.7f}     node2 voltage = {store_voltage[i]:.6f}     current = {store_current[i]:.6f}")

    


    plot_transient(store_time, store_voltage, name="transient_RC")



    





    # select output node for adjoint input
    '''output_node_for_sensitivity = 2
    sensitivities, std_dev = do_sensitivity_analysis(lu, VI, output_node_for_sensitivity, node_map, total_dim, w=w)
    print("Sensitivities of components:")
    for name, sens in sensitivities.items():
        print(f"{name}: {sens:.4f} V/unit")
    print(f"output voltage V = {VI[node_map[output_node_for_sensitivity]]} $\pm$ {std_dev} V")

    # # Example Bode Plot
    # run_bode_plot(test_directory + "ac_lowpass.txt", output_node=2, start_freq=10, stop_freq=100000, points=200, name = "lowpass")
    # run_bode_plot(test_directory + "ac_resonance.txt", output_node=3, start_freq=10, stop_freq=100000, points=200, name = "resonance")

    # Example ac sensitivity
    # print("do sweep")
    plot_sensitivity_sweep(components, output_node_for_sensitivity, "C1", start_f=10, end_f=1000, name="lowpass_C1_sensitivity")

    
'''