# Judge claude-opus-5.5, shards 029-034

Start: 2026-09-29 00:44 (system clock). End: 2026-09-29 00:53 (system clock).

Systematic difficulties:
- Suspected reference errors (reference_suspect=true, candidate judged on the mathematics): J0816 (problem says 11:00 PM; reference uses 11 AM), J0831 (a 10% speed increase gives 400/11 s; the reference cuts the time by 10%), J0853 (reference uses 4+1 apples per day), J0869 (reference solution accepts x=-2 and x=1, but its final answer lists only 1).
- The problem text contradicts itself ("ten stalls" vs "twenty stalls"): J0759, J0832. I followed the reference's 10-stall reading and gave low confidence.
- Ambiguous word problems where the candidate's literal reading differs from the reference: J0750 (half of what is left), J0766 (linear vs compound depreciation), J0777 (does travel cost include supplies), J0829 ("three times more"), J0837 ("cheapest lumber"), J0854 (magazines = issues), J0739 (loss boxed as -16).
- Figure-dependent items read from Asymptote code: J0738, J0758, J0798, J0810, J0843, J0862.
- Pedantic first-error choices: J0735 (wrong half-angle identity but a correct conclusion), J0743 (false claim m≠n that is never used), J0790 (false period 8, correct answer), J0795 (triangle-inequality simplification before the counting error), J0874 (restriction at step 20; I placed the error at the exhaustiveness claim in step 40).
