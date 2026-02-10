import numpy as np

def generate_nodal_equations(filename):
    components = [] # store circuit elements 
    max_node = 0 # maximum node in the netlist 

    # parse the netlist
    with open(filename, "r") as file:
        for line in file:
            line = line.strip() 
            if not line or line.startswith('*'): 
                continue
                
            parts = line.split()
            comp_name = parts[0].upper()
            comp_type = comp_name[0]    
            n1 = int(parts[1]) #from node
            n2 = int(parts[2]) #to node
            val = float(parts[3])
            
            components.append({
                'name': comp_name,  
                'type': comp_type, 
                'n1': n1, 
                'n2': n2, 
                'value': val
            })
            max_node = max(max_node, n1, n2)

    # initialize the matrix
    Y = np.zeros((max_node, max_node)) #exluding ground node
    J = np.zeros(max_node)

    # start stamping
    for comp in components:
        n1, n2, val = comp['n1'], comp['n2'], comp['value']
        
        # adjust for 0-based indexing (Node 1 --> Index 0)
        i, j = n1 - 1, n2 - 1
        
        if comp['type'] == 'R':
            g = 1.0 / val  # R to G
            
            # where in the matrix to stamp g value
            if n1 > 0:
                Y[i, i] += g
            if n2 > 0:
                Y[j, j] += g
            if n1 > 0 and n2 > 0:
                Y[i, j] -= g
                Y[j, i] -= g
                
        elif comp['type'] == 'I':
            # flip signs since J moves to RHS
            if n1 > 0:
                J[i] -= val
            if n2 > 0:
                J[j] += val
                
    return Y, J, max_node

if __name__ == "__main__":
    Y_result, J_result, node_count = generate_nodal_equations("netlist.txt")

    # output
    print(f"--- NODAL ANALYSIS SYSTEM (Nodes: 1 to {node_count}) ---")
    print("\n[Y] Matrix (Admittance):")
    print(Y_result)
    print("\n[J] Vector (Sources):")
    print(J_result)