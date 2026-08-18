# 📊 Infrastructure-as-Code Agent Benchmarks: Comprehensive Evaluation

This document provides a quantitative, data-driven analysis of the performance across the four iterative architectures of the RAG-based IaC Agent: **Basic**, **Standard RAG**, **Advanced RAG**, and **Secure RAG**.

The benchmark evaluated **16 test scenarios** (4 workflows × 4 complex IaC prompts):
1. **3-Tier VPC Topology**: VPC with 1 public & 2 private subnets, web EC2, app EC2, and RDS database.
2. **EKS Cluster + IAM Trust Policy**: EKS cluster with dedicated IAM assume role policy and `AmazonEKSClusterPolicy`.
3. **Terraform Remote-State Backend**: S3 bucket (versioning, AES-256 encryption, public access block) + DynamoDB table with `LockID` hash key.
4. **Mixed EC2 Fleet**: VPC with public/private subnets, Launch Templates mixing 5 On-Demand and 4 Spot instances.

---

## 1. Executive Summary

- **🥇 Undisputed Leader — Secure RAG**: Achieved a **100% (4/4) validity rate** and the highest code quality score (**4.5 / 5.0**). Its Checkov security verification loop caught critical edge cases (such as the EC2 Fleet configuration in Prompt 4) and self-healed them within 2 retries.
- **🥈 Advanced RAG**: Scored **3.75 / 5.0** with superior context compression (~3,793 characters) and smart reranking, though it hit retry limits on the intricate EC2 Fleet IAM role syntax.
- **🥉 Standard RAG**: Achieved **3.5 / 5.0** but suffered from prompt dilution (~4,534 characters) and generated a syntax error in the S3 state backend.
- **❌ Basic Workflow**: Produced catastrophic hallucinations on 2 prompts (generating `random_pet` and `local_file` dummy resources instead of AWS EKS/EC2 Fleets), resulting in two **1/5** scores.

---

## 2. Quantitative Results Breakdown

| Metric | Basic | Standard RAG | Advanced RAG | Secure RAG |
| :--- | :--- | :--- | :--- | :--- |
| **Valid Code Rate** | 75.0% (3/4) | 75.0% (3/4) | **100.0% (4/4)** | **100.0% (4/4)** |
| **Average Quality Score** (out of 5) | 3.00 / 5.0 | 3.50 / 5.0 | 3.50 / 5.0 | **4.50 / 5.0** 🥇 |
| **Avg. Execution Latency** | **96.3s** | 131.2s | **104.4s** | 138.9s |
| **Avg. Self-Healing Retries** | 1.25 | 1.00 | **0.50** | 0.75 |
| **Avg. Context Size** | 0 chars | ~4,534 chars | ~3,755 chars | ~3,866 chars |

---

## 3. Scenario-by-Scenario Matrix

| Prompt | Basic Score (Valid) | Standard RAG (Valid) | Advanced RAG (Valid) | Secure RAG (Valid) |
| :--- | :--- | :--- | :--- | :--- |
| **1. 3-Tier VPC Topology** | 5/5 (❌ Retried 3x) | **5/5** (✅ Pass) | 4/5 (✅ Pass) | 4/5 (✅ Pass) |
| **2. EKS Cluster + IAM** | 1/5 (✅ Hallucinated) | **5/5** (✅ Pass) | 4/5 (✅ Pass) | **5/5** (✅ Pass) |
| **3. S3 + DynamoDB Backend** | **5/5** (✅ Pass) | 2/5 (❌ Syntax Err) | **5/5** (✅ Pass) | **5/5** (✅ Pass) |
| **4. Mixed EC2 Fleet** | 1/5 (✅ Hallucinated) | 2/5 (✅ Bad Config) | 1/5 (✅ Passed Syn, 1 Retry) | **4/5** (✅ Healed 2x) |

---

## 4. Key Architectural Insights

### 1. The Catastrophic Hallucination Risk in LLM-Only Workflows
Without grounded RAG context, the base LLM took shortcuts when faced with unfamiliar or complex AWS resources (EKS IAM trust syntax and EC2 Fleet configs), hallucinating non-AWS local/pet resources to pass superficial compiler checks.

### 2. Context Precision vs Context Volume
Standard RAG injected the largest context payloads (~4,534 chars), but more text did not equate to better code (3.5 vs 3.75). Advanced RAG's CrossEncoder reranker trimmed ~740 characters of noise while boosting semantic relevance.

### 3. Checkov-Assisted Self-Healing Wins the Day
On Prompt 4 (Mixed EC2 Fleet), all other workflows either hallucinated, misconfigured the Spot capacity, or failed syntax validation. Secure RAG used Checkov's strict static analysis feedback in the Fixer Node to repair the configuration within 2 retry cycles, successfully reaching a deployable state with 4/5 quality.

