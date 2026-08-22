"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import { PipelineVisualizer, Stage } from "@/components/PipelineVisualizer/PipelineVisualizer";
import { TrustScoreCard, TrustData } from "@/components/TrustScoreCard/TrustScoreCard";
import { TerraformViewer } from "@/components/TerraformViewer/TerraformViewer";
import { HitLPanel } from "@/components/HitLPanel/HitLPanel";
import { LogTerminal, LogEntry } from "@/components/LogTerminal/LogTerminal";
import { useSSEStream, SSEEvent } from "@/lib/sse";
import { submitHitLAction, getHealth } from "@/lib/api";
import {
  Zap, Send, RotateCcw, ChevronRight,
  Server, Database, Shield, GitBranch, Cpu, Upload
} from "lucide-react";
import styles from "./page.module.css";

const WORKFLOW_STAGES: Record<string, Stage[]> = {
  basic:    [
    { id: "Architect_Node",       icon: "🏗️", label: "Architect",  status: "idle" },
    { id: "Validator_Node",       icon: "🔎", label: "Validator",  status: "idle" },
    { id: "Fixer_Node",           icon: "🔧", label: "Fixer",      status: "idle" },
  ],
  rag: [
    { id: "Retriever_Node",       icon: "🔍", label: "Retriever",  status: "idle" },
    { id: "Architect_Node",       icon: "🏗️", label: "Architect",  status: "idle" },
    { id: "Validator_Node",       icon: "🔎", label: "Validator",  status: "idle" },
    { id: "Fixer_Node",           icon: "🔧", label: "Fixer",      status: "idle" },
  ],
  advanced: [
    { id: "Retriever_Node",       icon: "🔍", label: "Retriever",  status: "idle" },
    { id: "Architect_Node",       icon: "🏗️", label: "Architect",  status: "idle" },
    { id: "Validator_Node",       icon: "🔎", label: "Validator",  status: "idle" },
    { id: "Fixer_Node",           icon: "🔧", label: "Fixer",      status: "idle" },
    { id: "Trust_Assessor_Node",  icon: "🛡️", label: "Trust",     status: "idle" },
  ],
  secure: [
    { id: "Retriever_Node",       icon: "🔍", label: "Retriever",  status: "idle" },
    { id: "Architect_Node",       icon: "🏗️", label: "Architect",  status: "idle" },
    { id: "Validator_Node",       icon: "🔎", label: "Validator",  status: "idle" },
    { id: "Fixer_Node",           icon: "🔧", label: "Fixer",      status: "idle" },
    { id: "Trust_Assessor_Node",  icon: "🛡️", label: "Trust",     status: "idle" },
  ],
  hitl: [
    { id: "Retriever_Node",       icon: "🔍", label: "Retriever",  status: "idle" },
    { id: "Architect_Node",       icon: "🏗️", label: "Architect",  status: "idle" },
    { id: "Validator_Node",       icon: "🔎", label: "Validator",  status: "idle" },
    { id: "Fixer_Node",           icon: "🔧", label: "Fixer",      status: "idle" },
    { id: "Plan_Node",            icon: "📋", label: "Plan",       status: "idle" },
    { id: "Trust_Assessor_Node",  icon: "🛡️", label: "Trust",     status: "idle" },
    { id: "HitL_Node",            icon: "⏸️", label: "Review",    status: "idle" },
    { id: "Apply_Node",           icon: "🚀", label: "Apply",      status: "idle" },
    { id: "Destroy_Node",         icon: "💥", label: "Destroy",    status: "idle" },
  ],
};

const EXAMPLE_PROMPTS = [
  {
    icon: <Server size={16} />,
    label: "3-Tier Web App",
    tag: "Multi-AZ",
    prompt: "Deploy a highly-available 3-tier architecture with a VPC, public subnet for a web EC2 instance, private subnet for an app EC2 instance, and an RDS MySQL database with multi-AZ.",
  },
  {
    icon: <Cpu size={16} />,
    label: "EKS Cluster + IAM",
    tag: "Kubernetes",
    prompt: "Provision an AWS EKS cluster with a dedicated IAM role using the AmazonEKSClusterPolicy, inside a new VPC with two subnets across different availability zones.",
  },
  {
    icon: <Database size={16} />,
    label: "Terraform Backend",
    tag: "Remote State",
    prompt: "Set up a Terraform remote state backend using an S3 bucket with versioning and AES-256 encryption, and a DynamoDB table with a LockID hash key for state locking.",
  },
];

const WORKFLOW_OPTIONS = [
  { id: "hitl",     label: "HitL RAG",     description: "Human-gated + Apply to AWS", badge: "Recommended" },
  { id: "advanced", label: "Advanced RAG", description: "CrossEncoder reranking",      badge: "Fast" },
  { id: "secure",   label: "Secure RAG",   description: "Checkov security scan",       badge: "" },
  { id: "basic",    label: "Basic",        description: "LLM only, no retrieval",      badge: "" },
];

function now() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

export default function DashboardPage() {
  const [workflow, setWorkflow] = useState("hitl");
  const [prompt, setPrompt]   = useState("");
  const [running, setRunning] = useState(false);
  const [chunkCount, setChunkCount] = useState<number | undefined>();
  const [stages, setStages]   = useState<Stage[]>(WORKFLOW_STAGES["hitl"]);
  const [logs, setLogs]       = useState<LogEntry[]>([]);
  const [files, setFiles]     = useState<Record<string, string>>({});
  const [trust, setTrust]     = useState<TrustData | null>(null);
  const [citations, setCitations] = useState<string[]>([]);
  const [hitlPaused, setHitlPaused] = useState(false);
  const [hitlLoading, setHitlLoading] = useState(false);
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [activePrompt, setActivePrompt] = useState("");
  const [costEstimate, setCostEstimate] = useState<number>(0);
  const [totalRetries, setTotalRetries] = useState(0);
  const [streamingCode, setStreamingCode] = useState("");
  const [planSummary, setPlanSummary]     = useState<Record<string, number>>({});
  const [blastOk, setBlastOk]             = useState(true);
  const [costOk, setCostOk]               = useState(true);
  const [applyStatus, setApplyStatus]     = useState("");
  const [showWorkflowMenu, setShowWorkflowMenu] = useState(false);
  const [sreMode, setSreMode] = useState(false);
  const [sreFiles, setSreFiles] = useState<Record<string, string>>({});
  const sreInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    getHealth().then((h) => setChunkCount(h.chunk_count)).catch(() => {});
  }, []);

  const addLog = useCallback((msg: string, level: LogEntry["level"] = "info") => {
    setLogs((prev) => [...prev.slice(-80), { ts: now(), msg, level }]);
  }, []);

  const updateStage = useCallback((nodeId: string, status: Stage["status"]) => {
    setStages((prev) => prev.map((s) => (s.id === nodeId ? { ...s, status } : s)));
  }, []);

  const handleEvent = useCallback((ev: SSEEvent) => {
    if (ev.event === "node_update") {
      updateStage(ev.node, ev.status as Stage["status"]);
      const levelMap: Record<string, LogEntry["level"]> = {
        running: "info", done: "ok", warning: "warn", failed: "err", paused: "warn",
      };
      addLog(`[${ev.node.replace("_Node", "")}] ${ev.status}`, levelMap[ev.status] ?? "info");
      
      const evt = ev as SSEEvent & { retry_count?: number, validation_errors?: string, docs_retrieved?: number };
      if (evt.docs_retrieved !== undefined && evt.docs_retrieved > 0) {
        addLog(`[Retriever] Retrieved ${evt.docs_retrieved} documents from knowledge base`, "ok");
      }
      if (evt.validation_errors && evt.validation_errors !== "Success") {
        addLog(`[Error Details] ${evt.validation_errors.split("\\n")[0]}...`, "err");
      }
      
      const rc = evt.retry_count;
      if (rc) setTotalRetries((p) => Math.max(p, rc));
    } else if (ev.event === "plan_preview") {
      const e = ev as SSEEvent & { plan_summary?: Record<string,number>; cost_estimate_monthly?: number; blast_radius_passed?: boolean; cost_ceiling_passed?: boolean };
      if (e.plan_summary) setPlanSummary(e.plan_summary);
      if (e.cost_estimate_monthly !== undefined) setCostEstimate(e.cost_estimate_monthly);
      if (e.blast_radius_passed !== undefined) setBlastOk(e.blast_radius_passed);
      if (e.cost_ceiling_passed !== undefined) setCostOk(e.cost_ceiling_passed);
      updateStage("Plan_Node", e.blast_radius_passed === false || e.cost_ceiling_passed === false ? "warning" : "done");
      addLog(`[Plan] ${e.plan_summary?.create ?? 0} create, ${e.plan_summary?.update ?? 0} update, ${e.plan_summary?.delete ?? 0} destroy · $${(e.cost_estimate_monthly ?? 0).toFixed(0)}/mo`, "ok");
    } else if (ev.event === "trust_score") {
      updateStage(ev.node, "done");
      setTrust({ score: ev.score, label: ev.label, factors: ev.factors, explanation: ev.explanation });
      addLog(`[Trust] ${ev.label} (${Math.round(ev.score * 100)}%)`, "ok");
    } else if (ev.event === "hitl_pause") {
      const e = ev as SSEEvent & { files?: Record<string, string>; citations?: string[] };
      if (e.files) setFiles(e.files);
      if (e.citations) setCitations(e.citations);
      updateStage("HitL_Node", "paused");
      setHitlPaused(true);
      addLog("[HitL] Workflow paused — awaiting human review", "warn");
    } else if (ev.event === "apply_result") {
      const e = ev as SSEEvent & { status?: string };
      setApplyStatus(e.status ?? "");
      updateStage("Apply_Node", e.status === "applied" ? "done" : "failed");
      addLog(`[Apply] ${e.status}`, e.status === "applied" ? "ok" : "err");
    } else if (ev.event === "destroy_result") {
      const e = ev as SSEEvent & { status?: string };
      updateStage("Destroy_Node", e.status === "destroyed" ? "done" : "failed");
      addLog(`[Destroy] ${e.status}`, e.status === "destroyed" ? "ok" : "err");
    } else if (ev.event === "complete") {
      setFiles(ev.files ?? {});
      setCitations(ev.citations ?? []);
      if (ev.trust_score && !trust) {
        setTrust({ score: ev.trust_score, label: ev.trust_label, factors: ev.trust_factors, explanation: ev.trust_explanation });
      }
      addLog("[Pipeline] Complete ✓", "ok");
    } else if (ev.event === "error") {
      addLog(`[Error] ${ev.message}`, "err");
    }
  }, [addLog, updateStage, trust]);

  const { start } = useSSEStream({ onEvent: handleEvent, onDone: () => setRunning(false) });

  function handleWorkflowChange(id: string) {
    setWorkflow(id);
    setStages(WORKFLOW_STAGES[id] ?? []);
    reset();
    setShowWorkflowMenu(false);
  }

  function reset() {
    setLogs([]); setFiles({}); setTrust(null); setCitations([]);
    setHitlPaused(false); setCostEstimate(0); setTotalRetries(0);
    setStreamingCode(""); setPlanSummary({}); setBlastOk(true);
    setCostOk(true); setApplyStatus("");
    setStages((prev) => prev.map((s) => ({ ...s, status: "idle" })));
  }

  async function handleSubmit() {
    if (running) return;
    if (sreMode) {
      // SRE mode: validate uploaded files
      if (Object.keys(sreFiles).length === 0) { addLog("[SRE] No files uploaded yet.", "warn"); return; }
      const tid = crypto.randomUUID();
      setThreadId(tid);
      setActivePrompt("[SRE Upload]");
      reset();
      setSreFiles({});
      setRunning(true);
      addLog(`[SRE Mode] Starting validation pipeline with ${Object.keys(sreFiles).length} file(s)…`, "info");
      // Submit as hitl with upload_mode
      await start("hitl", "__sre_upload__", tid);
      return;
    }
    if (!prompt.trim()) return;
    const tid = crypto.randomUUID();
    setThreadId(tid);
    setActivePrompt(prompt);
    reset();
    setRunning(true);
    addLog(`Starting ${workflow} pipeline…`, "dim");
    setPrompt("");
    await start(workflow, prompt, tid);
  }

  async function handleHitL(
    action: "approve" | "patch" | "apply" | "destroy",
    patchRequest = "",
    overrideConfirmed = false,
  ) {
    setHitlLoading(true);
    try {
      const res = await submitHitLAction(threadId, workflow, action, patchRequest, activePrompt, overrideConfirmed);
      if (action === "approve") {
        setHitlPaused(false);
        updateStage("HitL_Node", "done");
        addLog("Approved — saved to history", "ok");
      } else if (action === "apply") {
        setHitlPaused(false);
        const status = (res as Record<string, string>)?.apply_status ?? "";
        setApplyStatus(status);
        updateStage("Apply_Node", status === "applied" ? "done" : "failed");
        addLog(`Apply ${status}`, status === "applied" ? "ok" : "err");
      } else if (action === "patch") {
        setHitlPaused(false);
        setRunning(true);
        reset();
        addLog("Patch requested — re-running pipeline…", "info");
        await start(workflow, patchRequest, threadId);
      }
    } finally {
      setHitlLoading(false);
    }
  }

  const showResult  = Object.keys(files).length > 0;
  const hasStarted  = running || logs.length > 0 || showResult;
  const currentWorkflow = WORKFLOW_OPTIONS.find((w) => w.id === workflow)!;

  return (
    <div className="app-layout">
      <Sidebar
        chunkCount={chunkCount}
        selectedWorkflow={workflow}
        onWorkflowChange={handleWorkflowChange}
      />

      <main className={styles.main}>
        {/* Top header bar */}
        <header className={styles.topbar}>
          <div className={styles.topbarLeft}>
            <div className={styles.workflowSelector} onClick={() => setShowWorkflowMenu(!showWorkflowMenu)}>
              <Zap size={13} style={{ color: "var(--blue)" }} />
              <span className={styles.workflowName}>{currentWorkflow.label}</span>
              <ChevronRight size={12} className={`${styles.chevron} ${showWorkflowMenu ? styles.chevronOpen : ""}`} />
            </div>

            {showWorkflowMenu && (
              <div className={styles.workflowMenu}>
                {WORKFLOW_OPTIONS.map((w) => (
                  <button
                    key={w.id}
                    className={`${styles.workflowOption} ${workflow === w.id ? styles.workflowOptionActive : ""}`}
                    onClick={() => handleWorkflowChange(w.id)}
                  >
                    <span className={styles.workflowOptionName}>{w.label}</span>
                    <span className={styles.workflowOptionDesc}>{w.description}</span>
                    {w.badge && <span className={`badge badge-blue ${styles.workflowBadge}`}>{w.badge}</span>}
                  </button>
                ))}
              </div>
            )}
          </div>

          {hasStarted && (
            <button className={`btn btn-ghost btn-sm ${styles.resetBtn}`} onClick={reset} disabled={running}>
              <RotateCcw size={13} />
              New Run
            </button>
          )}

          {/* SRE Mode toggle */}
          <button
            className={`btn btn-ghost btn-sm ${sreMode ? styles.sreModeActive : ""}`}
            onClick={() => { setSreMode(!sreMode); reset(); }}
            title="SRE Mode: upload existing Terraform files for validation"
            style={{ marginLeft: "0.5rem", color: sreMode ? "var(--yellow)" : "var(--text-dim)",
              border: sreMode ? "1px solid rgba(245,158,11,0.4)" : undefined }}
          >
            <Upload size={13} />
            {sreMode ? "SRE Mode ON" : "SRE Mode"}
          </button>
        </header>

        {/* Hero — only shown when nothing is started */}
        {!hasStarted && (
          <div className={styles.hero}>
            <div className={styles.heroBadge}>
              <Shield size={11} />
              Self-healing agentic RAG
            </div>
            <h1 className={`gradient-text ${styles.heroTitle}`}>
              Build Infrastructure,<br />Not Config Files.
            </h1>
            <p className={styles.heroSub}>
              Describe what you need. TerraForge generates production-ready, security-hardened Terraform — grounded in 15K+ AWS docs, validated deterministically, and gated by human approval before it touches your account.
            </p>
          </div>
        )}

        {/* Pipeline Visualizer */}
        {stages.length > 0 && <PipelineVisualizer stages={stages} />}

        {/* Log Terminal */}
        <LogTerminal entries={logs} />

        {/* Example prompts — only when idle */}
        {!hasStarted && (
          <div className={styles.examples}>
            <div className={styles.examplesLabel}>Try an example</div>
            <div className={styles.examplesGrid}>
              {EXAMPLE_PROMPTS.map((ex) => (
                <button
                  key={ex.label}
                  className={styles.exampleCard}
                  onClick={() => setPrompt(ex.prompt)}
                >
                  <div className={styles.exampleCardTop}>
                    <span className={styles.exampleIcon}>{ex.icon}</span>
                    <span className={styles.exampleTag}>{ex.tag}</span>
                  </div>
                  <div className={styles.exampleLabel}>{ex.label}</div>
                  <div className={styles.examplePrompt}>{ex.prompt}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Streaming preview */}
        {streamingCode && !showResult && !hitlPaused && (
          <div className="animate-pulse">
            <TerraformViewer files={{ "generating...": streamingCode + " ▋" }} />
          </div>
        )}

        {/* HitL panel + code when paused */}
        {hitlPaused && (
          <>
            <TerraformViewer files={files} />
            {trust && <TrustScoreCard data={trust} />}
            <HitLPanel
              threadId={threadId}
              workflow={workflow}
              prompt={activePrompt}
              files={files}
              planSummary={planSummary}
              costEstimate={costEstimate}
              blastRadiusPassed={blastOk}
              costCeilingPassed={costOk}
              onAction={handleHitL}
              loading={hitlLoading}
              applyStatus={applyStatus}
            />
          </>
        )}

        {/* Final result */}
        {showResult && !hitlPaused && (
          <>
            {trust && <TrustScoreCard data={trust} />}
            <TerraformViewer files={files} />

            {/* Run metadata row */}
            <div className={styles.metaRow}>
              {costEstimate > 0 && (
                <span className="badge badge-purple">💰 ~${costEstimate.toFixed(0)}/month estimated</span>
              )}
              {totalRetries === 0 && (
                <span className="badge badge-green">✅ First-try generation</span>
              )}
              {totalRetries > 0 && (
                <span className="badge badge-yellow">🔧 {totalRetries} self-heal {totalRetries === 1 ? "retry" : "retries"}</span>
              )}
              {applyStatus === "applied" && (
                <span className="badge badge-green">🚀 Applied to AWS</span>
              )}
            </div>

            {/* Citations */}
            {citations.length > 0 && (
              <div className={styles.citations}>
                <div className={styles.citationsLabel}>📚 Sources</div>
                <div className={styles.citationsList}>
                  {citations.map((c, i) => (
                    <span key={i} className="badge badge-blue">{c}</span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Input area ── */}
        <div className={styles.inputZone}>
          <div className={styles.inputCard}>
            <textarea
              ref={textareaRef}
              className={styles.textarea}
              placeholder="Describe the AWS infrastructure you want to build… e.g. 'Deploy a highly-available 3-tier web app with RDS'"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
              }}
              rows={1}
              disabled={running || hitlPaused}
            />
            <div className={styles.inputFooter}>
              <button
                className={`btn btn-primary ${styles.sendBtn}`}
                onClick={handleSubmit}
                disabled={running || hitlPaused || (!prompt.trim() && !sreMode)}
              >
                {running ? (
                  <><span className="spinner" style={{ width: 13, height: 13 }} /> Running…</>
                ) : (
                  <><Send size={13} /> Generate</>
                )}
              </button>
            </div>
          </div>
          <div className={styles.inputHintRow}>⏎ Generate · ⇧⏎ New line</div>
        </div>
      </main>
    </div>
  );
}
