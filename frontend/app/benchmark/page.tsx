"use client";

import Link from "next/link";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import { ChevronLeft, BarChart2 } from "lucide-react";
import styles from "./page.module.css";

// ── Static data from benchmark_results.json ──────────────────────────────────

const TIERS = [
  {
    id: "basic",
    label: "Basic LLM",
    color: "#8b949e",
    colorBg: "rgba(139,148,158,0.1)",
    passRate: 75,
    passFraction: "3 / 4",
    avgScore: 3.0,
    avgLatency: 96.3,
    minLatency: 57.4,
    maxLatency: 199.6,
    avgRetries: 1.25,
    avgContext: 0,
    icon: "⚡",
    tagline: "No retrieval — pure pre-trained memory",
  },
  {
    id: "rag",
    label: "Standard RAG",
    color: "#58a6ff",
    colorBg: "rgba(88,166,255,0.1)",
    passRate: 75,
    passFraction: "3 / 4",
    avgScore: 3.5,
    avgLatency: 131.2,
    minLatency: 74.5,
    maxLatency: 227.4,
    avgRetries: 1.0,
    avgContext: 4534,
    icon: "🔍",
    tagline: "Naive cosine search — context dilution risk",
  },
  {
    id: "advanced",
    label: "Advanced RAG",
    color: "#bc8cff",
    colorBg: "rgba(188,140,255,0.1)",
    passRate: 100,
    passFraction: "4 / 4",
    avgScore: 3.5,
    avgLatency: 104.4,
    minLatency: 92.6,
    maxLatency: 129.5,
    avgRetries: 0.5,
    avgContext: 3755,
    icon: "🎯",
    tagline: "CrossEncoder reranker + MultiQuery expansion",
  },
  {
    id: "secure",
    label: "Secure RAG",
    color: "#3fb950",
    colorBg: "rgba(63,185,80,0.1)",
    passRate: 100,
    passFraction: "4 / 4",
    avgScore: 4.5,
    avgLatency: 138.9,
    minLatency: 70.4,
    maxLatency: 215.5,
    avgRetries: 0.75,
    avgContext: 3866,
    icon: "🛡️",
    tagline: "Checkov CIS scan loop + self-healing",
  },
];

const SCENARIOS = [
  {
    label: "3-Tier VPC Topology",
    sub: "VPC + subnets + EC2 + RDS",
    scores: [
      { score: 5, valid: false, note: "❌ Retried 3×" },
      { score: 5, valid: true,  note: "✅ Pass" },
      { score: 4, valid: true,  note: "✅ Pass" },
      { score: 4, valid: true,  note: "✅ Pass" },
    ],
  },
  {
    label: "EKS Cluster + IAM",
    sub: "EKS + trust policy + AmazonEKSClusterPolicy",
    scores: [
      { score: 1, valid: true,  note: "✅ Hallucinated" },
      { score: 5, valid: true,  note: "✅ Pass" },
      { score: 4, valid: true,  note: "✅ Pass" },
      { score: 5, valid: true,  note: "✅ Pass" },
    ],
  },
  {
    label: "S3 + DynamoDB Backend",
    sub: "Versioning, AES-256, LockID table",
    scores: [
      { score: 5, valid: true,  note: "✅ Pass" },
      { score: 2, valid: false, note: "❌ Syntax error" },
      { score: 5, valid: true,  note: "✅ Pass" },
      { score: 5, valid: true,  note: "✅ Pass" },
    ],
  },
  {
    label: "Mixed EC2 Fleet",
    sub: "Launch Template: 5 On-Demand + 4 Spot",
    scores: [
      { score: 1, valid: true,  note: "✅ Hallucinated" },
      { score: 2, valid: true,  note: "✅ Bad config" },
      { score: 1, valid: true,  note: "✅ Incomplete" },
      { score: 4, valid: true,  note: "✅ Healed 2×" },
    ],
  },
];

const V2_RESULTS = [
  { tier: "Naive LLM",     result: false, score: 1, tokens: 58556, color: "#ff7b72" },
  { tier: "Standard RAG",  result: false, score: 2, tokens: 37345, color: "#e3b341" },
  { tier: "Advanced RAG",  result: true,  score: 5, tokens: 7859,  color: "#3fb950" },
];

function scoreColor(s: number) {
  if (s >= 5) return "#3fb950";
  if (s >= 4) return "#58a6ff";
  if (s >= 3) return "#e3b341";
  return "#ff7b72";
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className={styles.barTrack}>
      <div className={styles.barFill} style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function BenchmarkPage() {
  const maxContext = Math.max(...TIERS.map(t => t.avgContext));
  const maxTokens = Math.max(...V2_RESULTS.map(r => r.tokens));

  return (
    <div className="app-layout">
      <Sidebar selectedWorkflow="hitl" onWorkflowChange={() => {}} />

      <main className={`main-content ${styles.main}`}>
        {/* Header */}
        <div className={styles.header}>
          <Link href="/" className={`btn btn-ghost`}>
            <ChevronLeft size={16} /> Back
          </Link>
          <div>
            <h1 className={styles.title}><BarChart2 size={22} /> Benchmark Results</h1>
            <p className={styles.subtitle}>
              16-scenario evaluation across 4 agentic tiers · LLM-as-Judge scoring · LangSmith telemetry
            </p>
          </div>
        </div>

        {/* Tier Cards */}
        <div className={styles.tierGrid}>
          {TIERS.map((t) => (
            <div key={t.id} className={styles.tierCard} style={{ borderColor: t.color + "44", background: t.colorBg }}>
              <div className={styles.tierTop}>
                <span className={styles.tierIcon}>{t.icon}</span>
                <div>
                  <div className={styles.tierLabel} style={{ color: t.color }}>{t.label}</div>
                  <div className={styles.tierTagline}>{t.tagline}</div>
                </div>
              </div>
              <div className={styles.tierStats}>
                <div className={styles.stat}>
                  <div className={styles.statVal}>{t.passFraction}</div>
                  <div className={styles.statKey}>Pass Rate</div>
                  <Bar pct={t.passRate} color={t.passRate === 100 ? "#3fb950" : "#e3b341"} />
                </div>
                <div className={styles.stat}>
                  <div className={styles.statVal} style={{ color: scoreColor(t.avgScore) }}>{t.avgScore.toFixed(1)}<span className={styles.statUnit}>/5</span></div>
                  <div className={styles.statKey}>Avg Quality</div>
                  <Bar pct={t.avgScore / 5 * 100} color={scoreColor(t.avgScore)} />
                </div>
                <div className={styles.stat}>
                  <div className={styles.statVal}>{t.avgLatency.toFixed(0)}<span className={styles.statUnit}>s</span></div>
                  <div className={styles.statKey}>Avg Latency</div>
                  <Bar pct={Math.min(t.avgLatency / 160 * 100, 100)} color="#bc8cff" />
                </div>
                <div className={styles.stat}>
                  <div className={styles.statVal}>{t.avgRetries.toFixed(2)}</div>
                  <div className={styles.statKey}>Avg Retries</div>
                  <Bar pct={t.avgRetries / 1.5 * 100} color="#ffa657" />
                </div>
              </div>
              {t.avgContext > 0 && (
                <div className={styles.contextRow}>
                  <span className={styles.contextLabel}>Avg Context Injected</span>
                  <span className={styles.contextVal}>{t.avgContext.toLocaleString()} chars</span>
                  <div className={styles.barTrack} style={{ marginTop: "0.3rem" }}>
                    <div className={styles.barFill} style={{ width: `${t.avgContext / maxContext * 100}%`, background: "#58a6ff" }} />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Key Insight */}
        <div className={styles.insightBox}>
          <span className={styles.insightIcon}>💡</span>
          <div>
            <strong>Counter-intuitive finding:</strong> Standard RAG had the same pass rate (75%) as the Naive LLM, and was 36s <em>slower</em> on average. Adding retrieval without a reranker caused context dilution — injecting irrelevant chunks that confused the LLM more than helped it. The CrossEncoder reranker is what pushed pass rate to 100%, with <strong>17% less context</strong>.
          </div>
        </div>

        {/* Scenario Matrix */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Scenario-by-Scenario Breakdown</h2>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Prompt</th>
                  {TIERS.map(t => (
                    <th key={t.id} style={{ color: t.color }}>{t.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SCENARIOS.map((scenario) => (
                  <tr key={scenario.label}>
                    <td>
                      <div className={styles.scenarioLabel}>{scenario.label}</div>
                      <div className={styles.scenarioSub}>{scenario.sub}</div>
                    </td>
                    {scenario.scores.map((s, i) => (
                      <td key={i} className={styles.scoreCell}>
                        <span className={styles.scoreNum} style={{ color: scoreColor(s.score) }}>
                          {s.score}/5
                        </span>
                        <span className={styles.scoreNote}>{s.note}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* V2 Proprietary */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>🔬 V2 — Proprietary Module Stress Test</h2>
          <p className={styles.sectionDesc}>
            Prompt: <em>"Deploy a 3-tier architecture using our internal <code>terraform-aws-acme-corp-vpc</code> module for networking"</em>
            <br />Zero documentation available to the LLM. Module schema injected only into the vector store.
          </p>
          <div className={styles.v2Grid}>
            {V2_RESULTS.map((r) => (
              <div key={r.tier} className={styles.v2Card} style={{ borderColor: r.color + "55" }}>
                <div className={styles.v2Header}>
                  <span className={styles.v2Tier}>{r.tier}</span>
                  <span className={styles.v2Result} style={{ color: r.result ? "#3fb950" : "#ff7b72" }}>
                    {r.result ? "✅ Passed" : "❌ Failed"}
                  </span>
                </div>
                <div className={styles.v2Stats}>
                  <div className={styles.v2Stat}>
                    <div className={styles.v2StatVal} style={{ color: scoreColor(r.score) }}>{r.score}/5</div>
                    <div className={styles.v2StatKey}>Quality</div>
                  </div>
                  <div className={styles.v2Stat}>
                    <div className={styles.v2StatVal}>{r.tokens.toLocaleString()}</div>
                    <div className={styles.v2StatKey}>Tokens Used</div>
                  </div>
                </div>
                <Bar pct={r.tokens / maxTokens * 100} color={r.color} />
                {r.tier === "Advanced RAG" && (
                  <div className={styles.v2Badge}>−86.6% tokens vs Naive · −73% latency · ~91s</div>
                )}
              </div>
            ))}
          </div>
          <div className={styles.insightBox} style={{ marginTop: "1rem" }}>
            <span className={styles.insightIcon}>🏆</span>
            <div>
              Advanced RAG reduced token cost by <strong>86.6%</strong> (58,556 → 7,859 tokens) and latency by <strong>73%</strong> on the proprietary module test — while being the <em>only</em> tier to achieve a deployable result with 5/5 quality.
            </div>
          </div>
        </section>

        {/* Methodology */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Evaluation Methodology</h2>
          <div className={styles.methodGrid}>
            <div className="card">
              <div className={styles.methodIcon}>🔧</div>
              <div className={styles.methodTitle}>Compiler Feedback Loop</div>
              <div className={styles.methodDesc}>Every generated architecture runs through <code>terraform init -backend=false && terraform validate</code>. Failures are fed back to the Fixer Node (up to 3 retries).</div>
            </div>
            <div className="card">
              <div className={styles.methodIcon}>⚖️</div>
              <div className={styles.methodTitle}>LLM-as-Judge</div>
              <div className={styles.methodDesc}><code>gemini-2.5-pro</code> at <code>temperature=0.0</code>, role-prompted as a HashiCorp Certified Terraform Associate. Scores 1–5 on security, intent, and standards adherence.</div>
            </div>
            <div className="card">
              <div className={styles.methodIcon}>📡</div>
              <div className={styles.methodTitle}>LangSmith Telemetry</div>
              <div className={styles.methodDesc}>P50/P99 latency, token consumption, and estimated API cost traced for every execution via LangSmith observability integration.</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
