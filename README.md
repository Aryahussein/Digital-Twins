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
8. 2/16 - Added ideal op amp - still linear MNA - to check with dc and ac analysis - works - tested with a simple non-inverting amp configuration.
9. 2/18 - Added transient solver - works! - verified using an RC circuit/RLC circuits and even an ideal op amp model.
10. 3/15 - Attempt 1 for transient adjoint sensitivity
11. 3/20 - Cleaned up the code base - made seperate models for res/cap/ind/sources
12. 3/21 - Implemented a simple diode model - non linear solver in the dc/ac/transient solvers
13. 3/22 - Verified the NR method for non linear solver - and used a half wave rectifer and a full wave rectifier to verify the reliability of the solver.