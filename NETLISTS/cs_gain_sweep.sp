* NMOS

* Supply
Vdd 1 0 DC 1.2

* Gate bias + AC input
Vg  2 0 DC 0.45

* Drain resistor (load)
Rd  1 3 80k
* NMOS
M1  3 2 4 0 nmos W=4u L=100n

Rs 4 0 1k

.dc Vd 0 1.2 0.01 SWEEP Vg 0.2 1.2 0.2
.print v 3
.end