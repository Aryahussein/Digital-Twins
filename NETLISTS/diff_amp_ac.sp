* Differential pair

* Supply
Vdd 3 0 1.2

* Inputs
* True differential inputs centred on Vcm = 0.55V
Vinp 1 0 SINE(0.44 0.001 1k 0)
Vinn 6 0 SINE(0.44 0.001 1k 180)

*Vinp 1 0 DC 0.44 AC 0.5
*Vinn 6 0 DC 0.44 AC 0.5 180

* NMOS diff pair
M1 2 1 4 0 nmos W=6u L=100n
M1 5 6 4 0 nmos W=6u L=100n

* Resistor load (Vbias raised from 0.8 to 0.5 to clear threshold)
Rl1 2 3 100K
R21 5 3 100k
*Cl1 2 0 1n
*Cl2 5 0 1n
R3l 4 0 0.01k
* I1  4 0 20u

.tran 1p 10n
* .dc Vinp 0 1.2 0.01
*.ac dec 20 1 100e6
*.diffgain 1 6 2 5
.print v 1 6
.print v 2 5