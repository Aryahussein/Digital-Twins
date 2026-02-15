# TO DO:
- VCCS: done
- AC impedances for capacitors and inductors (so just include jwL and 1/jwC in the admittance matrix)
- tests need to be written for individual modules to try and break the code: for example see 2.6 in the book "When do nodal equations fail?"
- Independent voltage source: "post-processing" but not sure how exactly
- AC nodal analysis: incorporate capacitors and inductors
- transient analysis
- Sensitivity
    - the sign used for current is opposite from the book's description; noted that this could potentially cause issues in the future
- MNA
    - the 1A & -1A current stamps should be scaled by orders of magnitude
# Nice to have
- Adjust indexing scheme to minimize off-diagonal elements: nodes that touch each other should be indexed with sequential node numbers
-> depth first search to determine ideal node indices?
- Allow nodes to have any name (not just 1, 2, etc.) to allow importing spice files directly from LTSpice (which uses N001, N002, etc. naming)




