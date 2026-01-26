import pandas as pd
import numpy as np

def generate_netlist_matrices(filename):
    """
    Reads a netlist CSV and returns the Y matrix, J vector, and Node Map.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None, None, None

    # 2. Identify Unique Nodes (excluding Ground)
    # Convert all node references to string to ensure "0" matches numeric 0
    df['In'] = df['In'].astype(str)
    df['Out'] = df['Out'].astype(str)
    
    ground_labels = {'0', 'GND', 'gnd'}
    
    # Get all unique nodes from both columns
    all_nodes = set(df['In']).union(set(df['Out']))
    
    # Filter out ground nodes and sort for consistent indexing
    system_nodes = sorted(list(all_nodes - ground_labels))
    
    # Create a mapping dictionary: Node Name -> Matrix Index
    # e.g., {'Vin': 0, 'A': 1, 'B': 2}
    node_map = {name: i for i, name in enumerate(system_nodes)}
    N = len(system_nodes)
    
    print(f"Processing Netlist: {filename}")
    print(f"Found {N} active nodes: {node_map}")
    
    # 3. Initialize Matrices
    Y = np.zeros((N, N))
    J = np.zeros((N, 1))

    # Helper function to get index (returns None if ground)
    def get_idx(node_name):
        if node_name in ground_labels:
            return None
        return node_map[node_name]

    # 4. Iterate and Stamp
    for _, row in df.iterrows():
        comp_type = row['Type']
        val = float(row['Value'])
        idx_in = get_idx(row['In'])
        idx_out = get_idx(row['Out'])
        
        if comp_type == "Resistor":
            # Conductance G = 1 / R
            g = 1.0 / val
            
            # Diagonal Elements (add G to self)
            if idx_in is not None:
                Y[idx_in, idx_in] += g
            if idx_out is not None:
                Y[idx_out, idx_out] += g
                
            # Off-Diagonal Elements (subtract G between nodes)
            if idx_in is not None and idx_out is not None:
                Y[idx_in, idx_out] -= g
                Y[idx_out, idx_in] -= g
                
        elif comp_type == "Current Source":
            # Current flows In -> Out
            # Current LEAVES 'In' node (Subtract from J)
            if idx_in is not None:
                J[idx_in] -= val
            
            # Current ENTERS 'Out' node (Add to J)
            if idx_out is not None:
                J[idx_out] += val

    # 5. Output Results
    print("\nAdmittance Matrix (Y):")
    print(Y)
    
    print("\nCurrent Vector (J):")
    print(J)
    
    return Y, J, node_map

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Replace 'your_netlist.csv' with your actual file path
    input_file = 'test_circuit.csv' 
    Y, J, nodes = generate_netlist_matrices(input_file)