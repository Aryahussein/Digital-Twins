Project for EE 7v88 - Digital Twins for IC Design & Beyond

Author - Miriyala Pranay Kamal

Progress -

1. 1/26 - main.py is the has the main code - it does extracts a netlist, filters out the comments, makes the nodal equations and creates the Y/V/I matrices.
2. 1/27 - added LU solver capability to check if the circuit has a voltage source (throws an error if it does), solves for the V matrix from Y/V matrices using the LU method and prints out the node voltages.