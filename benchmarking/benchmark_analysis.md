# 📊 Infrastructure-as-Code Agent Benchmarks: Unbiased Analysis

This document provides an objective, data-driven analysis of the performance across the four iterative architectures of the RAG-based IaC Agent: **Basic**, **Standard RAG**, **Advanced RAG**, and **Secure RAG**.

The benchmarks were executed across a 3-prompt dataset ranging from standard AWS infrastructure to complex compliance-heavy configurations (e.g., Transit Gateways, IRSA-enabled EKS, and WAFv2 Web ACLs).

---

## 1. Executive Summary

- **Most Reliable**: `Secure RAG` achieved a perfect 5/5 score across all prompts with a 100% validity rate, proving essential for enterprise-grade deployments, though it is the slowest architecture.
- **Best Balanced**: `Advanced RAG` drastically improved upon basic retrieval mechanisms, capturing a 100% validity rate while maintaining an average execution time under 3 minutes.
- **Fastest (but brittle)**: The `Basic` workflow is fast but struggles heavily with complex architectures, ultimately failing to generate valid code on more intricate instructions.

---

## 2. Quantitative Results Breakdown

| Metric | Basic | Standard RAG | Advanced RAG | Secure RAG |
| :--- | :--- | :--- | :--- | :--- |
| **Valid Code Rate** | 66.6% (2/3) | 66.6% (2/3) | **100% (3/3)** | **100% (3/3)** |
| **Code Quality Score** (out of 5) | 3.3 | 2.6 | 3.6 | **5.0** |
| **Avg. Latency** | **131.9s** | 144.6s | 180.4s | 279.9s |
| **Avg. Self-Healing Retries** | 1.3 | 1.6 | 1.3 | 1.6 |
| **Avg. Context Size** | 0 chars | ~4,918 chars | ~3,538 chars | ~3,455 chars |

> [!NOTE]
> *Token Usage and LLM API Call Costs were omitted from this analysis due to an active limitation in the upstream `langchain-google-vertexai` library not populating `usage_metadata` for the current model payload.*

---

## 3. Workflow Insights & Trade-offs

### Basic Workflow
* **Strengths:** Speed. With an average latency of ~132 seconds, it is the fastest option since it relies solely on the LLM's parametric memory without external retrieval.
* **Weaknesses:** It is highly brittle. On complex instructions, it hallucinates outdated Terraform patterns and lacks the necessary context to pass validation (failing the 3rd test case after 3 retries). 

### Standard RAG
* **Strengths:** Injects a massive amount of documentation (~4,918 characters average) into the prompt, giving the agent a higher ceiling for complex tasks.
* **Weaknesses:** Demonstrates the "lost in the middle" problem. The massive context injection actually resulted in the lowest average code quality score (2.6/5). The agent struggled to filter out irrelevant information, proving that *more* data is not always *better* data.

### Advanced RAG
* **Strengths:** The introduction of the `ms-marco` CrossEncoder Reranker slashed the context size by nearly 30% (down to ~3,538 characters) while simultaneously pushing the validity rate to a perfect 100%. By providing fewer, highly relevant snippets, the LLM hallucinated less and successfully generated deployable infrastructure every time.
* **Trade-offs:** The overhead of the Reranker node and the more complex multi-query generation adds about 35-45 seconds of latency compared to the basic workflows.

### Secure RAG
* **Strengths:** Unrivaled code quality and security. By integrating `checkov` into the Fixer Node's feedback loop, the agent actively patched vulnerabilities (e.g., unencrypted EBS volumes, open 0.0.0.0/0 ingress) before finalizing the code. This resulted in a flawless 5/5 judge score across the board.
* **Trade-offs:** This workflow is highly compute-intensive. Execution times averaged nearly 5 minutes (279.9s). The strict security policies meant the Fixer Node had to execute more retry loops (1.6 avg) to appease Checkov's hard-fail thresholds.

> [!IMPORTANT]
> **Architectural Recommendation**
> For internal development or staging environments, **Advanced RAG** provides the optimal balance of speed and reliability. For production environments, **Secure RAG** is strictly necessary, trading latency for automated DevSecOps compliance.
