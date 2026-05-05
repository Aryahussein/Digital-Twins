* NMOS AC small-signal (common source)

* Supply
Vdd 1 0 DC 1.2

* Gate bias + AC input
Vg  2 0 DC 0.45 AC 1

* Drain resistor (load)
Rd  1 3 80k
*Cd  3 0 1n
* NMOS
M1  3 2 0 0 nmos W=4u L=100n

.ac dec 200 1 1e10
.print v 3
.end