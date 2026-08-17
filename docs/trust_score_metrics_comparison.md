# Trust Score System: Pre-Fix vs. Post-Fix Metrics

## The Bugs We Fixed
Before our recent updates, the Trust Score wasn't getting the right numbers. It was stuck at around 50% for two main reasons:

1. **The Reranker Bug (Langchain's Fault):** The library we used to find the best documents was actually calculating perfect scores, but it had a bug where it would literally throw the scores in the trash before giving the documents back to us! Because our system couldn't see the scores, it assumed the score was "0", which mathematically translated to exactly 50%.
2. **The Database "Distance" Quirks:** Our database (ChromaDB) was trying to measure how close two documents were by using physical "distance" (like measuring with a ruler). Our system tried to convert this distance into a percentage, but the math formula was clunky and almost always gave a score of ~51%, even for great matches.

## The Simple Solutions
1. **Saving the Reranker Score:** We wrote a tiny piece of custom code (the `ScorePreservingReranker`) that intercepts the score and saves it securely *before* the library throws it away. Now our system can finally see the true 95% confidence scores!
2. **Switching the Database to "Similarity":** We changed a setting in the database to tell it to measure "Cosine Similarity" (true percentage) instead of "distance." We then rebuilt the database so it could give us real, accurate numbers like 85% or 92%.

---

## Metrics Comparison (Identical Prompt)
*We tested the identical "Route 53 zone association" prompt on the HitL RAG workflow.*

| Metric | Pre-Fix | Post-Fix | Delta |
|--------|---------|----------|-------|
| **Final Trust Score** | 🟡 65% (Review Recommended) | 🟢 86% (High Trust) | **+21%** |
| **Retrieval Similarity** | 51% | 65% | **+14%** |
| **Reranker Confidence** | 50% | 95% | **+45%** |
| **Validation Pass** | 100% | 100% | - |
| **Raw Reranker Logit** | `0.000` (Discarded) | `3.111` (Captured) | - |

> [!TIP]
> Notice how the **Reranker Confidence** skyrocketed from a static 50% to a hyper-confident 95%. This proves the pipeline was actually finding highly relevant infrastructure patterns the entire time, but the mathematical bugs were suppressing the true score!
