* NMOS Id vs Vds (family for different Vgs)

* Drain voltage (this will be swept → Vds)
Vd   1 0 DC 0

* Gate bias (this will be stepped manually via inner sweep)
Vg   2 0 DC 0.7

* NMOS
M1   1 2 3 0 nmos W=4u L=100n

* Small source resistor (helps convergence)
Rs   3 0 1

* Sweep Vds
.dc Vd 0 1.2 0.01
.dc Vd 0 1.2 0.1

* Print drain current
.print i Vd

.end