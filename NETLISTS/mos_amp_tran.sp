* NMOS AC small-signal (common source)

* Supply
Vdd 1 0 DC 1.2

* Gate bias + AC input
Vinp 2 0 SINE(0.45 0.001 1k 0)

* Drain resistor (load)
Rd  1 3 80k
*Cd  3 0 1n
* NMOS
M1  3 2 0 0 nmos W=4u L=100n

.tran 100u 100m
.print v 2
.print v 3
.end