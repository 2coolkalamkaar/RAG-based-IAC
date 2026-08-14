# 🧠 Interview Prep: Terraform Architect Agent (RAG-based IaC)

> Your go-to guide to confidently present this project live in an interview. Covers the narrative, technical depth, and Q&A you need to impress.

---

## 🎯 The 30-Second Elevator Pitch

> *"I built an autonomous IaC generation agent that solves a real problem: general-purpose LLMs hallucinate Terraform syntax, generate insecure configs, and produce code that can't actually deploy. My tool takes a plain-English request, retrieves grounded context from the official AWS Terraform provider documentation, generates multi-file production-grade Terraform code with hard security rules enforced, then autonomously validates it with the actual Terraform CLI and TFLint, self-corrects on failure, and finally estimates the monthly cloud cost using Infracost — all in a single workflow."*

---

## 🔥 The Core Problem Statement

**Why does this problem matter?**

- General-purpose LLMs (GPT-4, Gemini) are trained on the general internet. Terraform HCL is a **domain-specific language** with very strict syntax and rapidly-evolving provider APIs.
- The AWS Terraform Provider has **1,584+ resource types**, each with specific required arguments, deprecated fields, and nested block structures that change between provider versions.
- LLMs **hallucinate** resource attributes, mix deprecated arguments, forget required blocks, and generate security anti-patterns like open SSH ports, unencrypted volumes, and public IP assignments.
- Most LLM-generated Terraform code **cannot `terraform validate`** on the first try.

**Your solution:** Ground the LLM with the *exact, current, official* provider documentation through a RAG pipeline, then close the loop with real validation tooling.

---

## 🏗️ The Three Workflows (Your Benchmark Progression)

This is your core narrative: you built the system **iteratively**, testing and benchmarking each version.

### Workflow 1: Basic (Baseline, `agent_workflow.py`)

**Architecture:** `User → Architect Node → Validator Node ⇄ Fixer Node`

- No external knowledge. Pure LLM reasoning.
- The Architect generates Terraform code from scratch.
- The Validator runs `terraform validate` + `tflint`.
- On failure, the Fixer sees the error and rewrites (up to 3 retries).
- **Context length:** 0 (no RAG).

**What it proves:** A zero-shot LLM agentic loop can produce valid syntax *eventually*, but it's slow (131s on hard queries), inconsistent (score: 2/5 on complex infra), and has no grounding.

---

### Workflow 2: Standard RAG (`agent_workflow_rag.py`)

**Architecture:** `User → Retriever Node → Architect Node → Validator Node ⇄ Fixer Node`

**New addition: Retriever Node**
- Uses **ChromaDB** (local vector store, 13,779 chunks of AWS provider docs).
- Uses **HuggingFace `all-MiniLM-L6-v2`** embeddings (local CPU, no API key).
- Does a simple **cosine similarity search** (`k=6 docs`).
- Injects retrieved docs into the Architect's system prompt.
- Also adds **LangGraph `MemorySaver`** for persistent in-session conversation memory.

**What it proves:** RAG adds factual grounding, but simple top-k retrieval fetches broad, noisy context. Sometimes retrieves irrelevant docs. Scored 1/5 on one test and failed completely (3 retries, invalid).

---

### Workflow 3: Advanced RAG (`agent_workflow_advanced_rag.py`) — Your Flagship

**Architecture:** `User → Advanced Retriever → Architect → Validator ⇄ Fixer → Cost Estimator`

**The Smart Retrieval Pipeline (3-stage):**

1. **MultiQuery Expansion:** `mq_llm` (Gemini 2.5 Pro, `temp=0.0`) generates 3-5 semantically diverse re-phrasings of the user query. This dramatically increases recall, catching documents that simple keyword matching would miss.
2. **CrossEncoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): After over-fetching 12 docs, the reranker scores each doc's *actual relevance* to the original query — much more precise than embedding similarity. Keeps top-5.
3. **Post-retrieval Python filter:** Excludes `iac_eval_dataset` chunks (evaluation data that snuck into the DB — caught and handled gracefully).

**Additional upgrades:**
- **Cost Estimator Node:** After validation passes, writes `.tf` files to a temp dir and runs `infracost breakdown` to get a monthly cost breakdown.
- **SqliteSaver memory:** Persistent conversation state across app restarts (stored in `state.db`).
- **Citation tracking:** Every doc's source path is collected and shown to the user.

---

## 📊 Benchmark Results (Your Data Story)

| Workflow | Test | Time (s) | Valid | Score | Retries | Context Tokens |
|---|---|---|---|---|---|---|
| **Basic** | EC2 Fleet | 62.8 | ✅ | 5/5 | 0 | 0 |
| **Basic** | Complex VPC | 132.0 | ✅ | 5/5 | 1 | 0 |
| **Basic** | Multi-region | 93.6 | ✅ | 2/5 | 1 | 0 |
| **Standard RAG** | EC2 Fleet | 73.2 | ✅ | 2/5 | 0 | 4,507 |
| **Standard RAG** | Complex VPC | 154.6 | ✅ | 2/5 | 2 | 4,199 |
| **Standard RAG** | Multi-region | 168.2 | ❌ | 1/5 | 3 | 5,142 |
| **Advanced RAG** | EC2 Fleet | 88.0 | ✅ | **5/5** | 0 | 3,353 |
| **Advanced RAG** | Complex VPC | 130.2 | ✅ | **5/5** | 1 | 3,182 |
| **Advanced RAG** | Multi-region | 64.1 | ✅ | **5/5** | 0 | 3,766 |

### Key Takeaways to Tell the Interviewer

- **Advanced RAG always validates (3/3), always scores 5/5** — the other two don't.
- **Standard RAG actually performed worse than Basic** — this is a real finding! Adding noisy RAG context can confuse the LLM. Precision beats recall.
- **Advanced RAG uses *fewer* context tokens than Standard RAG** (avg ~3,434 vs ~4,616) — because the CrossEncoder threw away irrelevant docs. Less token usage = lower cost per request.
- **Advanced RAG is the fastest on average** — with precise context, the LLM produces correct code faster (fewer retries).

---

## 🛠️ Technical Deep-Dives (For Technical Interviewers)

### The ETL Pipeline (`etl_pipeline.py`)

> *"My knowledge base is always fresh. I built a custom incremental ETL pipeline."*

- **Source:** Official `terraform-provider-aws` GitHub repo (`website/docs/r/*.html.markdown`)
- **Hashing:** `SHA-256` hash of each file. On re-run, only files whose hash changed are re-indexed — no duplicates, no full rebuilds needed.
- **Splitter:** `RecursiveCharacterTextSplitter` with `chunk_size=1000, overlap=200`. Splits on `##`/`###` headers to preserve resource block context.
- **Scale:** 1,584 docs → 13,779 chunks
- **Three modes:** `--dry-run` (preview), `--full-rebuild` (wipe + reindex), default (incremental)

### The Validator (`validate_terraform_code` function)

> *"This is the heart of the self-healing loop."*

1. Writes `.tf` files to a **temp directory** (isolated, cleaned up after).
2. `terraform init -backend=false` — initializes providers without touching remote state.
3. `terraform validate` — checks syntax and internal consistency.
4. `tflint --minimum-failure-severity=error` — runs security lint rules (AWS-specific plugin). Catches open SSH, missing encryption, etc.
5. Returns `(bool, str)` tuple. On failure, the error string is fed directly to the Fixer Node as context.

### The Agentic State Machine (LangGraph `StateGraph`)

> *"This isn't a linear chain — it's a stateful graph with conditional routing."*

- **`AgentState` TypedDict** carries all inter-node data: messages, retrieved_context, terraform_code, validation_errors, retry_count, citations, cost_estimate.
- **Conditional edge** on `Validator_Node`: Routes to `Fixer_Node` on failure, `Cost_Estimator_Node` on success, or `END` when max retries (3) is hit.
- **Self-healing loop:** The `retry_count` in state prevents infinite loops. Max 3 fixer attempts.

### Why Local Embeddings?

> *"I deliberately chose not to use a cloud embedding API."*

- `all-MiniLM-L6-v2` runs entirely on CPU, no API key, zero per-request cost.
- For a domain-specific knowledge base that rarely changes, quality of embeddings matters less than the precision of retrieval — which is what the CrossEncoder reranker solves.
- This keeps the embedding cost $0 while the LLM handles the expensive reasoning.

---

## 💡 Design Decisions You Made (And Why)

| Decision | Rationale |
|---|---|
| **3 workflows, not 1** | Iterative benchmarking — proves the Advanced RAG is genuinely better with data, not just assumption |
| **CrossEncoder over Bi-Encoder for reranking** | Bi-encoders compare query and doc independently. CrossEncoder sees them *together* — much more accurate relevance scoring, at the cost of compute |
| **`temp=0.0` for MultiQuery LLM** | Need deterministic, semantically diverse query generation. Low temperature = consistent alternative phrasings |
| **`temp=0.2` for Architect LLM** | Slight creativity for code structure, but mostly deterministic for correctness |
| **SqliteSaver over MemorySaver** | SqliteSaver persists to disk — conversation state survives server restarts. MemorySaver is in-memory only |
| **No `terraform plan`** | Requires live AWS credentials. `validate` + `tflint` catches 95% of errors without needing cloud auth |
| **HCL file parsing (2-pattern regex)** | LLMs format filenames inconsistently — inside the code block or as a markdown header. Two-pattern approach handles both |

---

## 🤔 Anticipated Interview Questions & Model Answers

### Q: "How is this different from just calling ChatGPT with a Terraform prompt?"

> "Three ways. First, grounding: I inject actual current provider docs, not the LLM's training data which may be months or years old. Second, validation: I run real `terraform validate` and `tflint`, not just asking the LLM to 'check' its own code. Third, autonomy: if validation fails, the system *automatically* rewrites the code using the specific error as context — no human in the loop. ChatGPT gives you text. This gives you deployable infrastructure."

---

### Q: "What's the difference between MultiQuery and standard RAG retrieval?"

> "Standard RAG encodes the user's single query into a vector and finds the top-k nearest docs. If the user writes 'I need a load balancer' and the docs say 'Application Load Balancer configuration', the embedding similarity might miss it. MultiQuery uses the LLM to rephrase the query into 3-5 semantically equivalent versions — 'ALB setup', 'aws_lb resource', 'load balancer Terraform block' — and retrieves for all of them, then merges results. This dramatically improves recall for domain-specific technical queries."

---

### Q: "What is a CrossEncoder and why is it better than embedding similarity for reranking?"

> "A bi-encoder (what you use for retrieval) encodes the query and document *separately* into vectors and measures distance. It's fast but loses the interaction between query and document. A CrossEncoder takes the query and document *together* as a single input and directly predicts a relevance score. It sees the full context of both simultaneously, making it far more precise — but too slow to run against the entire corpus, so you only run it on the top-12 pre-retrieved candidates."

---

### Q: "Why did Standard RAG perform *worse* than the Basic workflow in your benchmark?"

> "This is my most interesting finding. Adding noisy, low-precision context can actually confuse the LLM. When the retriever fetches 6 generic docs that aren't specifically relevant to the query, the LLM tries to follow them and generates incorrect or overly verbose code that breaks validation. The Advanced RAG's CrossEncoder throws away the irrelevant docs, leaving only the 5 most precise ones. Precision over recall — quality of context matters more than quantity. This is why the Advanced RAG used *fewer* tokens but scored better."

---

### Q: "How does the self-healing loop work exactly?"

> "When `terraform validate` or `tflint` fails, the error message is stored in the `AgentState` as `validation_errors`. The `routing_edge` conditional function reads the state and routes to the `Fixer_Node` instead of END. The Fixer receives both the broken code and the specific error message and regenerates the complete corrected files. The new code goes back to the Validator. This cycle can repeat up to 3 times — after that, it exits to prevent infinite loops. In my benchmarks, most fixes happened on the first retry."

---

### Q: "What's in your vector database and how did you build it?"

> "13,779 chunks from 1,584 official AWS Terraform provider documentation files, scraped directly from the HashiCorp GitHub repo. I built a custom ETL pipeline that does SHA-256 hashing of each file — on updates, only changed or new files are re-indexed, making it incremental rather than full rebuilds. Each chunk is 1,000 tokens with 200-token overlap, split on Markdown section headers to preserve resource block context. The embeddings use HuggingFace's all-MiniLM-L6-v2 locally — zero API cost."

---

### Q: "Why LangGraph instead of a simple LangChain chain?"

> "LangChain chains are linear — good for question-answering pipelines. LangGraph models the workflow as a directed graph with nodes, edges, and state. I need conditional routing (validator → fixer OR cost estimator), cycles (the fix-validate loop), and persistent state (retry_count, accumulated messages, terraform_code). LangGraph handles all of this natively. The `StateGraph` with conditional edges is exactly the right abstraction for an autonomous self-correcting loop."

---

### Q: "How do you handle the cost of running this?"

> "By design, the most expensive components are minimized or free. Embeddings run locally (zero cost). The ChromaDB vector store is local (zero cost). The MultiQuery expansion and Architect calls are the only LLM costs. Using Gemini 2.5 Flash (which I've now migrated to via Vertex AI) costs $0.075 per 1M tokens — a typical request with ~4,000 input tokens costs under $0.001. The CrossEncoder runs locally on CPU. Infracost is free for open source. Total operational cost per request is fractions of a cent."

---

### Q: "What would you improve or scale next?"

> "Several things. First, async production mode — I have `agent_workflow_advanced_rag_prod.py` which uses Python `asyncio`. Second, CI/CD integration — expose the workflow as a REST API endpoint that gets triggered on pull requests to validate IaC changes automatically. Third, support for other cloud providers (GCP, Azure) by building equivalent vector stores from their Terraform providers. Fourth, multi-agent evaluation — add an LLM judge that scores the generated code's architectural quality beyond just syntax validity."

---

## 🎤 Live Demo Flow (If They Ask You to Show It)

1. **Open Streamlit UI** → `http://localhost:8501`
2. **Select "Advanced RAG" workflow** from the sidebar
3. **Type:** `"Create a secure VPC with public and private subnets, an Application Load Balancer, and an EC2 instance in the private subnet with SSM access for remote management."`
4. **Walk through tabs as they load:**
   - **Architect Notes tab** → "This is the LLM's reasoning and design rationale"
   - **Terraform Code tab** → "Three files: main.tf, variables.tf, outputs.tf — all generated and validated"
   - **Citations tab** → "These are the exact provider doc files used to ground the code"
   - **Cost Analysis tab** → "Infracost's estimate before you deploy a single resource"
5. **Point out the green banner:** `✨ Production Ready — Code passed all syntax & security validations`

---

## 🔑 Key Numbers to Memorize

| Metric | Value |
|---|---|
| Vector DB size | 13,779 chunks from 1,584 AWS resource docs |
| Embedding model | `all-MiniLM-L6-v2` (local, zero cost) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini 2.5 Pro via Google Vertex AI |
| Max retries | 3 |
| Advanced RAG score | **5/5 on all 3 benchmark tests** |
| Context reduction vs Standard RAG | ~25% fewer tokens (4,616 → 3,434 avg) |
| Advanced RAG validity | **100% (3/3)** vs Standard RAG **67% (2/3)** |

---

*Good luck! You built something genuinely impressive — make sure you own it in the room.*
