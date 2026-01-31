# TO DO:
- VCCS: done
- AC impedances for capacitors and inductors (so just include jwL and 1/jwC in the admittance matrix)
- tests need to be written for individual modules to try and break the code
- Independent voltage source: "post-processing" but not sure how exactly
- AC nodal analysis: incorporate capacitors and inductors
- transient analysis
# Nice to have
- Adjust indexing scheme to minimize off-diagonal elements: nodes that touch each other should be indexed with sequential node numbers
-> depth first search to determine ideal node indices?
- Allow nodes to have any name (not just 1, 2, etc.) to allow importing spice files directly from LTSpice (which uses N001, N002, etc. naming)


