* RC lowpass

* Supply
Vdd 1 0 AC 1

Rl 1 2 100k
C1 2 0 10u

.ac dec 20 1m 1000k
.print v 2
.end