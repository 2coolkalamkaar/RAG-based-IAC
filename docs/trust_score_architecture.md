# AI Trust Score Architecture & Feature Release

This document serves as a comprehensive record of the development, architectural decisions, and implementation details for the **AI Trust Score** feature shipped for the RAG-based IaC pipelines.

## 🌟 Feature Overview

We introduced a quantitative and qualitative **Trust Score** mechanism to evaluate the reliability and security of the AI-generated Terraform code. Rather than relying on a "black box" generation, the pipeline now provides a transparent confidence metric (`0-100%`) alongside a human-readable tier label (`🟢 High Trust`, `🟡 Review Recommended`, `🔴 Low Trust`) and a natural language explanation of the score.

This feature was successfully rolled out to:
1. **Advanced RAG** (3-Factor Formula)
2. **Secure RAG** (4-Factor Formula)

---

## 🏛️ Architectural Decisions & Rationale

During the development phase, several key design decisions were made to balance accuracy, transparency, and latency.

### 1. Zero-Latency Telemetry Extraction
**Challenge:** Calculating trust scores usually requires running separate evaluator LLM calls or re-running similarity searches, adding significant latency to the pipeline.
**Decision:** We opted for a **Zero-Latency Data Exhaust** approach. 
- **Reranker Scores:** We tapped directly into the `doc.metadata["relevance_score"]` that was already being calculated by the CrossEncoder during the retrieval phase. This provided high-fidelity semantic confidence at exactly 0ms overhead.
- **Retrieval Similarity:** We swapped the standard retriever query for `vector_store.similarity_search_with_relevance_scores()`. Because the local ChromaDB call doesn't require a network request, it adds negligible latency (~50ms) while providing exact cosine-similarity bounds.

### 2. Node Graph Placement
**Challenge:** Where should the trust score be calculated in the LangGraph?
**Decision:** We implemented the `Trust_Assessor_Node` as a terminal gateway just before the `END` state. 
- **Why:** By placing it after the `Validator ↔ Fixer` loop and the `Cost_Estimator`, the trust node has complete visibility into the final state of the pipeline (including how many retries were needed and if validation ultimately succeeded or failed). 
- **Routing:** Both the "Valid" path and the "Max Retries Reached" path route through the Trust Assessor, ensuring no execution escapes without a trust badge.

### 3. Metric Calculations & Normalization
**Challenge:** How do we convert raw machine learning outputs (which are mathematically unbounded or context-dependent) into standardized 0-1 metrics?
**Decision:** We established a strict mathematical pipeline for each factor:
- **Retrieval Similarity:** We use standard **Cosine Similarity** directly from the ChromaDB vector math. Cosine similarity naturally bounds between `-1` and `1`. Since relevant documents will fall in the positive domain, we map this directly to a `0.0 - 1.0` scale. A high cosine similarity indicates the user's prompt strongly matches existing Terraform provider documentation.
- **Reranker Confidence:** The MS-MARCO CrossEncoder model outputs "logits" (raw, unnormalized prediction values) which typically range from roughly `-10` to `+10`. We applied a mathematical **Sigmoid Function** (`1 / (1 + e^-x)`) to squash these unbounded logits into a smooth `0.0 - 1.0` probability curve. This represents the model's confidence that the retrieved document actually answers the query.

### 4. Weighted Algorithm + Rule-Based Caps
**Challenge:** How do we combine disparate metrics (ML probabilities vs. boolean validation checks) into a single human-readable score?
**Decision:** We used a deterministic, weighted linear formula rather than an LLM prompt to ensure the score is mathematically reproducible and perfectly stable.
- **Rule-Based Caps:** We implemented hard overrides. If syntax validation fails or Checkov detects a critical vulnerability, the final score is artificially capped at `40%` (`🔴 Low Trust`), ensuring that unsafe code never accidentally receives a "High Trust" rating just because the retrieval score was high.

---

## 🛠️ Implementation Details

### The Scoring Formulas & Weighting Logic

The weighting logic was explicitly designed to balance the **source grounding** (the RAG component) against the **execution safety** (the Infrastructure-as-Code component).

**Advanced RAG (3-Factor)**
*   📚 **Retrieval Similarity (35%)**: Measures the relevance of the vector search. Weighted heavily because RAG performance dictates whether the Architect hallucinates.
*   🎯 **Reranker Confidence (35%)**: Measures the cross-encoder's verification of the retrieved context. Given equal weight to retrieval because a bad retrieval filtered well is better than a bad retrieval injected blindly.
*   ✅ **Validation Pass (30%)**: Terraform syntax and TFLint status. Weighted at 30% because while valid syntax is required, valid code that hallucinates wrong infrastructure is still dangerous. (Note: Validation failure overrides the entire score down to 40%).

**Secure RAG (4-Factor)**
When Checkov is introduced, the weights are adjusted to carve out a 20% penalty strictly for shift-left security checks.
*   📚 **Retrieval Similarity (25%)**
*   🎯 **Reranker Confidence (25%)**
*   ✅ **Validation Pass (30%)**: Kept at 30% to maintain a strong baseline for syntactical correctness.
*   🛡️ **Security Scan (20%)**: Checkov static analysis. If Checkov finds a critical issue, this drops to 0%, heavily tanking the score, and triggering the 40% hard-cap override.

### The UI & "Why this score?" Engine
We heavily modified the Streamlit UI (`multi_workflow_ui.py`) to render a beautiful, dynamic Trust Card.
- **Dynamic Factor Loading:** The UI automatically detects if `checkov_passed` exists in the graph state, seamlessly swapping between 3-factor and 4-factor progress bars.
- **HTML Bug Fix:** We encountered an issue where Streamlit's Markdown parser was converting our custom HTML progress bars into code blocks. We resolved this by bypassing the markdown engine and using pure string concatenation fed into `st.html()`.
- **Explainability:** We built a zero-latency `_build_trust_explanation` string-generator that translates the raw metrics into a highly readable paragraph (e.g., *"The knowledge base had moderate coverage... The code required 1 fix attempt(s) before passing validation..."*).

---

## 📂 Files Modified

1. **`workflows/agent_workflow_advanced_rag.py`**
   - Expanded `AgentState` with `trust_score`, `trust_label`, `trust_factors`, `trust_explanation`, `avg_retrieval_similarity`, `avg_reranker_score`.
   - Updated `retriever_node` to capture telemetry.
   - Built `trust_assessor_node` and wired it into `workflow.add_node()` and `workflow.add_edge()`.
2. **`workflows/agent_workflow_secure_rag.py`**
   - Mirrored the Advanced RAG updates but specifically integrated the `checkov_passed` extraction logic from the `validation_errors` string.
3. **`ui/multi_workflow_ui.py`**
   - Added the `Trust_Assessor_Node` to the graph metadata block for both workflows.
   - Built the custom HTML/CSS for the Trust Score dashboard.
   - Appended `trust_explanation` into the initialization states.
