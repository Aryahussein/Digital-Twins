* Full-wave bridge rectifier

V1 1 0 SINE(0 1 1k)

* Bridge diodes
D1 1 2
D2 0 2
D3 3 1
D4 3 0

* Load
R1 2 3 1k

.tran 1u 5m
.print v 1
.print v 2
.print v 3
.end