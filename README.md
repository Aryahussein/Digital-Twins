Project for EE 7v88 - Digital Twins for IC Design & Beyond

Author - Miriyala Pranay Kamal

Progress -

1. 1/26 - main.py is the has the main code - it does extracts a netlist, filters out the comments, makes the nodal equations and creates the Y/V/I matrices.
2. 1/27 - added LU solver capability to check if the circuit has a voltage source (throws an error if it does), solves for the V matrix from Y/V matrices using the LU method and prints out the node voltages.
3. 1/30 - Added Voltage source solving capability by using the same LU decomposition technique - checked with a voltage divider circuit
4. 1/31 - Checked with multiple sources (current+votlage) in the same circuit to check compatibility - works!
5. 2/1 - Added VCCS source
6. 2/4 - Added 2 new files - dc_sovler.py and ac_solver.py which you can choose like you do on the spice netlists: .op - for dc and .ac for ac analysis.
7. 2/5 - made a main.py which reads netlist -> and picks the solver based on the netlist. Plots bode plots using matplotlib for ac solver.