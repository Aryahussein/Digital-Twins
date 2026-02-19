V1 1 0 STEP(0 1 0)
R1 2 0 10k
R2 3 2 10k
O1 1 2 3 1e5

.tran 1e-5 0.01
.print v 3
.end
