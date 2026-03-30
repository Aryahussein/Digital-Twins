* Differential pair

* Supply
Vdd 3 0 1.2

* Inputs
* True differential inputs centred on Vcm = 0.55V
Vinp 1 0 SINE(0.7 0.01 1g 0)
Vinn 6 0 SINE(0.7 0.01 1g 180)

* NMOS diff pair
M1 2 1 4 0 nmos
M1 5 6 4 0 nmos

* Resistor load (Vbias raised from 0.8 to 0.5 to clear threshold)
Rl1 2 3 100k
R21 5 3 100k
R3l 4 0 50k

.tran 1p 10n
.print v 1 6
.print v 2 5