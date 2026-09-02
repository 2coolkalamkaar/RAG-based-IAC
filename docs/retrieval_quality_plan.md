# Retrieval Quality — Plan

> Companion to `docs/performance_optimizations.md`, which is about *speed* (making retrieval
> fast). This doc is about *correctness* — making sure the chunks the system pulls out of the
> knowledge base are actually the right ones. A fast wrong answer is still wrong.

---

## Why this is a separate problem from speed

Making retrieval faster (the singleton caching in `performance_optimizations.md`) doesn't change
*what* gets retrieved — it's the same chunks, just fetched quicker. Retrieval **quality** is a
different question entirely: out of everything in the knowledge base, did the system actually pull
the handful of documents that were truly relevant to this specific request? A system can be
instant and still be feeding the Architect the wrong context, which shows up downstream as vague
generated code, missed best practices, or citations that don't really justify what was built.

---

## How retrieval works today (so the gaps below make sense)

1. User's request goes through **MultiQuery expansion** — an LLM rewrites it into 3 similar
   questions to search from different angles.
2. Each rewritten question does a **semantic (embedding) search** against ChromaDB — it finds
   chunks whose *meaning* is close to the question, using vector math, not exact word matching.
3. The combined results go through a **CrossEncoder reranker** — a second, more careful model that
   re-scores each candidate chunk against the original question and keeps the top 5.
4. Those 5 chunks become the `### CONTEXT FROM SMART SEARCH ###` section of the Architect's prompt.

Every approach below targets a specific weak point in that four-step chain.

---

## Approach 1 — Hybrid search: add exact keyword matching alongside meaning-based search

**In simple words:** embedding search understands *meaning*, not *exact spelling*. Ask about
"S3 bucket versioning" and it'll usually find the right doc. But Terraform is full of precise,
literal identifiers — `aws_s3_bucket_versioning`, `server_side_encryption_configuration` — and a
meaning-based search sometimes drifts past the doc that has the *exact* term in favor of one that's
merely *related*. A keyword search (the classic "does this word literally appear" approach, e.g.
BM25) doesn't have that problem — but it also doesn't understand meaning at all. Combining both
covers each other's blind spot.

**How it would work:** run a keyword search (a `BM25Retriever` over the same chunk texts) alongside
the existing vector search, then merge and re-weight the two result lists (LangChain's
`EnsembleRetriever` does this merge — you give each retriever a weight, e.g. 60% semantic / 40%
keyword, and it combines the rankings).

**Effort:** Medium — needs a BM25 index built over the existing chunks (in-memory, from what's
already in ChromaDB) and one new retriever wired into the existing pipeline.

---

## Approach 2 — Filter by resource type before searching, using metadata that already exists

**In simple words:** every chunk in the knowledge base already carries a `resource_name` tag
(e.g. `s3_bucket`) — the same field the citation feature (Phase 5) uses to match generated code
back to its source doc. Right now that tag is only used *after* generation, for citations. It could
also be used *before* generation, to narrow the search: if the user's request clearly says
"S3 bucket," don't just hope the semantic search finds the S3 docs — actively filter down to
chunks tagged with a matching resource name first, *then* rank what's left.

**How it would work:** extract likely resource-type keywords from the user's request (simple
pattern matching against known AWS service names, or let the MultiQuery LLM tag the request with
resource types it's asking about), then pass a metadata `where` filter into the Chroma query
alongside the semantic search.

**Effort:** Low — the metadata already exists (built for Phase 5), this is just a new way to use it.

---

## Approach 3 — Wire in the policy-doc knowledge base (still unused)

**In simple words:** already flagged in `pipeline_review.md` and `pipeline_optimization_plan.md` —
6 policy markdown files (tagging rules, encryption requirements, IAM restrictions, etc.) sit in
`knowledge_base/policy_docs/` but nothing retrieves from them. Right now the Architect's hardcoded
prompt rules are the *only* place those policies live. Retrieving from a second, separate
collection means policies can be updated by editing a markdown file, and — more importantly for
retrieval quality — the Architect gets to *see* the specific policy chunk that justifies a rule,
not just a blanket instruction.

**How it would work:** index `knowledge_base/policy_docs/` into its own ChromaDB collection
(separate from the AWS provider docs), retrieve from both collections on every request, and label
which collection each chunk came from in the prompt (`[AWS DOCS]` vs `[ORG POLICY]`) so the
Architect — and eventually the reviewer, via citations — can tell "how do I do this" apart from
"am I allowed to do this."

**Effort:** Medium — new collection, new ETL step, prompt template change to merge both result sets.

---

## Approach 4 — A reranker actually trained for code/config docs, not web search

**In simple words:** the current reranker (`ms-marco-MiniLM-L-6-v2`) was trained on web search
click data — "does this webpage answer this Google-style question." Terraform documentation isn't
shaped like a webpage; it's structured reference material with argument names, types, and defaults.
A reranker trained on technical/code documentation (or at minimum, a bigger general-purpose
cross-encoder) would likely score relevance more accurately for this specific domain.

**How it would work:** swap the model name in `_get_cross_encoder()` and A/B test retrieval quality
before/after on a fixed set of test prompts (see Approach 7 for how to actually measure that
rather than eyeballing it).

**Effort:** Low to try, Medium to properly validate the swap was actually better and not just
different.

---

## Approach 5 — Diversity-aware selection (MMR) instead of pure top-N by score

**In simple words:** right now the top 5 chunks are just the 5 highest-scoring ones. If the
knowledge base has 3 near-duplicate chunks saying almost the same thing about the same resource,
all 3 can end up in the top 5, crowding out a genuinely different but slightly-lower-scoring chunk
about a different part of the request. MMR ("Maximal Marginal Relevance") picks results that are
both relevant *and* different from what's already been picked — so the 5 final chunks cover more
distinct ground instead of repeating each other.

**How it would work:** LangChain's Chroma retriever supports this out of the box —
`vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 12, "lambda_mult": 0.5})`
replaces the current plain similarity retriever with an MMR-aware one, with `lambda_mult`
controlling the relevance-vs-diversity trade-off.

**Effort:** Low — mostly a parameter change, worth testing what `lambda_mult` value works best.

---

## Approach 6 — Smarter chunking of the source documents

**In simple words:** how documents get cut into chunks matters as much as how they get searched.
A chunk that's cut off mid-example, or that mixes two unrelated resource types together because
the cut point landed in the wrong place, will retrieve poorly and read poorly even if it does get
picked. `data/etl_pipeline.py` already tries to split on resource block boundaries rather than
blindly by character count — worth revisiting whether that boundary detection is working well
across the whole doc set, and whether chunk size is tuned right (too small loses context, too
large dilutes relevance and wastes token budget).

**How it would work:** audit a sample of actual chunks in `chroma_db_terraform` for
mid-example cuts or mixed-resource chunks, then adjust the splitter's logic in `etl_pipeline.py`
accordingly. This is inspection-and-tuning work, not a new mechanism.

**Effort:** Medium — requires manually reviewing chunk boundaries, not just changing a parameter.

---

## Approach 7 — Actually measure retrieval quality, not just the final code

**In simple words:** this is the most important one, because it's what tells you whether any of
Approaches 1–6 actually helped. Right now the only signal available is "did the final generated
Terraform pass validation" — that conflates two very different things: good retrieval, and the LLM
just already knowing the answer without needing the retrieved context at all. A prompt can produce
great code with terrible retrieval (the model knew it anyway) or mediocre code with great retrieval
(the model didn't use the context well). Without separating those, you can't tell if a retrieval
change actually helped.

**How it would work:** extend the existing IaC-Eval benchmark with a small labeled set — for each
benchmark prompt, note which specific doc chunk(s) *should* be retrieved. Then measure standard
retrieval metrics against that: **recall@k** (was the right chunk in the top k at all?) and
**MRR** (how high up was it ranked?). This becomes the yardstick for every other approach on this
list — a way to say "Approach 4 improved recall@5 from 60% to 78%" instead of "it feels better."

**Effort:** Medium to set up once, then cheap to re-run for every future retrieval change — this
is the approach that makes every other approach here actually verifiable.

---

## Approach 8 — Query rewriting tuned for Terraform/AWS vocabulary specifically

**In simple words:** MultiQuery expansion already rewrites the user's request into 3 similar
questions before searching — but it uses a generic rewriting prompt, not one that knows it's
searching Terraform/AWS documentation specifically. A rewriting prompt that's told to produce
variants using actual HCL/AWS terminology (resource type names, argument names) alongside the
plain-English version would likely search better than 3 generically-phrased variants.

**How it would work:** replace the default `MultiQueryRetriever.from_llm(...)` prompt with a
custom one specifically instructing the rewriting LLM to include both a natural-language variant
and a Terraform-syntax-aware variant of the question.

**Effort:** Low — a prompt template change, testable quickly against the eval set from Approach 7.

---

## Suggested order

Given no eval harness exists yet, doing anything else first is flying blind — you won't be able to
tell if a change helped or just felt different.

1. **Approach 7 (measurement)** — build this first. Everything else is a guess without it.
2. **Approach 5 (MMR)** and **Approach 8 (better query rewriting)** — cheapest to try, easy to
   measure against Approach 7 once it exists.
3. **Approach 2 (metadata filtering)** — cheap, uses data that already exists from Phase 5.
4. **Approach 3 (policy-doc KB)** — already flagged elsewhere as the strongest differentiation
   story, not purely a quality fix but belongs on this list too.
5. **Approach 1 (hybrid search)** and **Approach 4 (better reranker)** — bigger effort, do these
   once the eval harness can actually confirm they're worth keeping.
6. **Approach 6 (chunking audit)** — manual, unglamorous, do it if the eval harness shows recall
   problems that the other fixes don't explain.
