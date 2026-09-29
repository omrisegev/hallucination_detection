# Judge fable-5.1, shards 000-004 (items K001-K118)

Start: 2026-09-29 ~17:35 (+03:00). End: 2026-09-29 18:00 (+03:00).

- Suspected reference errors (reference_suspect=true, candidate judged on the mathematics): K011 (reference says "54/6 = 9 groups", contradicting the stated six groups); K013, K058 (reference applies 21% of the original price each year, not a constant rate); K031, K099 (reference takes Lylah as 30% less than Adrien instead of Adrien/1.3); K068 (reference halves Ezra's total including his extra 150 books).
- Ambiguous problem wording where I followed the reference but with low confidence: K033/K101 ("half of what is left is sold equally"), K043/K093 (1.5 ft spacing vs 1 ft plant + 1.5 ft gap), K062 (60 g per feeding vs per cat), K075 ("cheapest lumber"), K020/K089 (problem says both "ten stalls" and "twenty stalls").
- Answer-form mismatches labeled as errors with low confidence: K023 (decimal 1.73 for sqrt3), K036 (gives 86 glasses and 42 plates, not the total 128). Signed/directional answers K057 (-25) and K059 (-3600, "decreases by 3600") were treated as equivalent to the reference.
- Correct final answers reached by unjustified restriction or lucky arithmetic: K015 (wrong algebra, right answer), K073/K085 (symmetry ansatz asserted as global maximum, labeled logic with low confidence), K004/K097 (ansatz with verification or forced symmetry, labeled no error).
- Harmless misstatements of the figure not counted as errors when nothing relied on them: K005 (centers K/O swapped), K014 step 3 (DE/EA labels), K083 step 11, K114 (A/B swapped).
- Awkward step splits: K087 (case analyses span step boundaries; first wrong total is in step 16), K118 (step 7 boxes 140 and then retracts to 50 in the same step; recorded as not matching).
- K030: the candidate's code has an off-by-one index bug and the claimed output (16) is fabricated; verified with Python that the code prints 24 and the true count is 25.
- K067: no final answer is stated (candidate_final_answer null).
