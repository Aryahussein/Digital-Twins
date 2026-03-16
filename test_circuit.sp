V1 1 0 STEP(0 1 1n)
R1 1 2 1k
C1 2 0 1n

.tran 1n 10u
.print v 2
.sens v(2)
.end