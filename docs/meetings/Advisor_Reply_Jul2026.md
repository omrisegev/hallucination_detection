# Reply to Ofir & Bracha (cc Amir) — Jul 2026

*Sent in reply to the Jul 1 2026 thread (Ofir's FUSE concern + Bracha's LR / cell / weights questions).*
*Attachments: `logistic_oracle3.png`, `lr_convergence.png`, `lr_weight_agreement.png` (pasted inline as `image.png`).*

---

On the FUSE paper - I agree It looks very similar: it's built on the same Nadler work (and theorem) we've been citing, the SML. The difference is the problem. FUSE picks the best of N candidate answers using a panel of separate verifier models. We detect whether a single answer is a hallucination, and we do it from one pass of one model, using its own internal probabilities, with no external verifiers. So it's the same tool on a different (though related) problem, plus our own advantage of one pass and one model.
Should we be concerned? I'd like to hear what you guys think about it. Let's set a meeting.

About The LR result -  After fixing an averaging mistake I made (one cell was included in the unsupervised average but not the supervised one), the supervised gap is a bit bigger than I sent:  4.7pp overall (5-feature LR 68.9% vs L-SML 64.2%) and +5pp on GPQA and RAG, while on reasoning both sit near the ceiling so the gap is still ~0.
logistic_oracle3.png
"5 features best" is mostly how the sets were picked - they aren't nested (the 9-set happens to drop spectral_entropy, a strong feature, so it's just a weaker subset). It was built previously according to what features don't tend to get saturated.  When I rank the features by auc and add them in order, CV is basically flat from 5 to 16 while the in-sample ceiling keeps climbing - so the extra features only overfit.
lr_convergence.png

Regarding the terms you asked about - by "cell" I meant one (dataset, model, temperature) test set. The "in-sample ceiling" is the LR fit and scored on the same data. I read  it measures the quality of the features, as an estimate of the best any linear rule on these features could reach.

The LR weights. They correlate only weakly with the L-SML weights. Both lean on the same top features (epr, spectral_entropy, cusum_max) but weigh them differently, so the two methods reach similar accuracy through different weightings.I guess it means the features are correlated enough that the exact weighting isn't unique.
image.png
I am continuing with the other action items we've talked about, but as I said, I want to hear what you think, and how we should continue.
Please tell me when it will be a good time for you next week.
Omri
