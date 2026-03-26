* Differential pair

* Supply
Vdd 3 0 1.2

* Inputs
* True differential inputs centred on Vcm = 0.55V
Vinp 1 0 SINE(0.9 0.01 1g)
Vinn 4 0 SINE(0.9 0.01 1g 180)

* Tail current source (was Rtail 5 0 2k — wrong, use current source)
Itail 0 5 100u

* NMOS diff pair
M1 6 1 5 0 nmos
M2 7 4 5 0 nmos

* PMOS load (Vbias raised from 0.8 to 0.5 to clear threshold)
Vbias 8 0 0.5

M3 6 8 3 3 pmos
M4 7 8 3 3 pmos

* Matched loads on both outputs
Rout  6 0 10k
Rout2 7 0 10k
Cload  6 0 10f
Cload2 7 0 10f

.tran 1p 10n
.print v 6 7
*.print v 7
.print v 1 4
*.print v 4
.end