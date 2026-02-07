def create_big_ladder(n_stages=500):
    lines = [
        "Vinput 1 0 10.0"
    ]
    
    for i in range(1, n_stages + 1):
        n_in = i
        n_out = i + 1
        # Series Resistor
        lines.append(f"R{i} {n_in} {n_out} 100")
        # Shunt Capacitor (to ground)
        lines.append(f"C{i} {n_out} 0 1e-6")
        
    with open("./testfiles/large_test.txt", "w") as f:
        f.write("\n".join(lines))

create_big_ladder(500)
