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
import { SendHorizonal } from "lucide-react";
import styles from "./page.module.css";

const WORKFLOW_STAGES: Record<string, Stage[]> = {
  basic:    [{ id: "Architect_Node", icon: "🏗️", label: "Architect",  status: "idle" }, { id: "Validator_Node", icon: "🔎", label: "Validator", status: "idle" }, { id: "Fixer_Node", icon: "🔧", label: "Fixer", status: "idle" }],
  rag:      [{ id: "Retriever_Node", icon: "🔍", label: "Retriever", status: "idle" }, { id: "Architect_Node", icon: "🏗️", label: "Architect", status: "idle" }, { id: "Validator_Node", icon: "🔎", label: "Validator", status: "idle" }, { id: "Fixer_Node", icon: "🔧", label: "Fixer", status: "idle" }],
  advanced: [{ id: "Retriever_Node", icon: "🔍", label: "Retriever", status: "idle" }, { id: "Architect_Node", icon: "🏗️", label: "Architect", status: "idle" }, { id: "Validator_Node", icon: "🔎", label: "Validator", status: "idle" }, { id: "Fixer_Node", icon: "🔧", label: "Fixer", status: "idle" }, { id: "Trust_Assessor_Node", icon: "🛡️", label: "Trust",  status: "idle" }],
  secure:   [{ id: "Retriever_Node", icon: "🔍", label: "Retriever", status: "idle" }, { id: "Architect_Node", icon: "🏗️", label: "Architect", status: "idle" }, { id: "Validator_Node", icon: "🔎", label: "Validator", status: "idle" }, { id: "Fixer_Node", icon: "🔧", label: "Fixer", status: "idle" }, { id: "Trust_Assessor_Node", icon: "🛡️", label: "Trust",  status: "idle" }],
  hitl:     [{ id: "Retriever_Node", icon: "🔍", label: "Retriever", status: "idle" }, { id: "Architect_Node", icon: "🏗️", label: "Architect", status: "idle" }, { id: "Validator_Node", icon: "🔎", label: "Validator", status: "idle" }, { id: "Fixer_Node", icon: "🔧", label: "Fixer", status: "idle" }, { id: "Trust_Assessor_Node", icon: "🛡️", label: "Trust",  status: "idle" }, { id: "HitL_Node", icon: "⏸️", label: "HitL", status: "idle" }, { id: "Patcher_Node", icon: "🔧", label: "Patcher", status: "idle" }],
};

function now() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

export default function DashboardPage() {
  const [workflow, setWorkflow] = useState("hitl");
  const [prompt, setPrompt] = useState("");
  const [running, setRunning] = useState(false);
  const [chunkCount, setChunkCount] = useState<number | undefined>();
  const [stages, setStages] = useState<Stage[]>(WORKFLOW_STAGES["hitl"]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [files, setFiles] = useState<Record<string, string>>({});
  const [trust, setTrust] = useState<TrustData | null>(null);
  const [citations, setCitations] = useState<string[]>([]);
  const [hitlPaused, setHitlPaused] = useState(false);
  const [hitlLoading, setHitlLoading] = useState(false);
  const [threadId, setThreadId] = useState<string>(() => crypto.randomUUID());
  const [activePrompt, setActivePrompt] = useState("");
  const [costEstimate, setCostEstimate] = useState("");
  const [totalRetries, setTotalRetries] = useState(0);
  const [streamingCode, setStreamingCode] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    getHealth().then((h) => setChunkCount(h.chunk_count)).catch(() => {});
  }, []);

  const addLog = useCallback((msg: string, level: LogEntry["level"] = "info") => {
    setLogs((prev) => [...prev.slice(-60), { ts: now(), msg, level }]);
  }, []);

  const updateStage = useCallback((nodeId: string, status: Stage["status"]) => {
    setStages((prev) =>
      prev.map((s) => (s.id === nodeId ? { ...s, status } : s))
    );
  }, []);


  const handleEvent = useCallback((ev: SSEEvent) => {
    if (ev.event === "node_update") {
      updateStage(ev.node, ev.status as Stage["status"]);
      const levelMap: Record<string, LogEntry["level"]> = {
        running: "info", done: "ok", warning: "warn", failed: "err", paused: "warn"
      };
      addLog(`${ev.node.replace("_Node","")}: ${ev.status}`, levelMap[ev.status] ?? "info");
      // Track retries from Fixer node events
      const retryCount = (ev as SSEEvent & { retry_count?: number }).retry_count;
      if (retryCount) setTotalRetries((prev: number) => Math.max(prev, retryCount));
    } else if (ev.event === "trust_score") {
      updateStage(ev.node, "done");
      setTrust({ score: ev.score, label: ev.label, factors: ev.factors, explanation: ev.explanation });
      addLog(`Trust Score: ${ev.label} (${Math.round(ev.score * 100)}%)`, "ok");
    } else if (ev.event === "hitl_pause") {
      updateStage("HitL_Node", "paused");
      setHitlPaused(true);
      addLog("Pipeline paused — awaiting human review", "warn");
    } else if (ev.event === "complete") {
      setFiles(ev.files ?? {});
      setCitations(ev.citations ?? []);
      if (ev.cost_estimate) setCostEstimate(ev.cost_estimate);
      if (ev.trust_score && !trust) {
        setTrust({ score: ev.trust_score, label: ev.trust_label, factors: ev.trust_factors, explanation: ev.trust_explanation });
      }
      addLog("Pipeline complete!", "ok");
    } else if (ev.event === "code_stream") {
      const chunk = (ev as SSEEvent & { chunk?: string }).chunk;
      if (chunk) setStreamingCode((prev) => prev + chunk);
    } else if (ev.event === "error") {
      addLog(`Error: ${ev.message}`, "err");
    }
  }, [addLog, updateStage, trust]);

  const { start } = useSSEStream({
    onEvent: handleEvent,
    onDone: () => setRunning(false),
  });


  function handleWorkflowChange(id: string) {
    setWorkflow(id);
    setStages(WORKFLOW_STAGES[id] ?? []);
    reset();
  }

  function reset() {
    setLogs([]);
    setFiles({});
    setTrust(null);
    setCitations([]);
    setHitlPaused(false);
    setCostEstimate("");
    setTotalRetries(0);
    setStreamingCode("");
    setStages((prev) => prev.map((s) => ({ ...s, status: "idle" })));
  }

  async function handleSubmit() {
    if (!prompt.trim() || running) return;
    const tid = crypto.randomUUID();
    setThreadId(tid);
    setActivePrompt(prompt);
    reset();
    setRunning(true);
    addLog(`Starting ${workflow} pipeline…`, "dim");
    setPrompt("");
    await start(workflow, prompt, tid);
  }

  async function handleHitL(action: "approve" | "patch", patchRequest = "") {
    setHitlLoading(true);
    try {
      await submitHitLAction(threadId, workflow, action, patchRequest, activePrompt);
      if (action === "approve") {
        setHitlPaused(false);
        updateStage("HitL_Node", "done");
        addLog("Approved — workflow complete!", "ok");
      } else {
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

  const showResult = Object.keys(files).length > 0;
  const hasStarted = running || logs.length > 0 || showResult;

  const EXAMPLE_PROMPTS = [
    { label: "3-Tier Web App", prompt: "Deploy a highly-available 3-tier architecture with a VPC, public subnet for a web EC2 instance, private subnet for an app EC2 instance, and an RDS MySQL database with multi-AZ." },
    { label: "EKS + IAM", prompt: "Provision an AWS EKS cluster with a dedicated IAM role using the AmazonEKSClusterPolicy, inside a new VPC with two subnets across different availability zones." },
    { label: "Terraform Remote State", prompt: "Set up a Terraform remote state backend using an S3 bucket with versioning and AES-256 encryption, and a DynamoDB table with a LockID hash key for state locking." },
  ];

  return (
    <div className="app-layout">
      <Sidebar
        selectedWorkflow={workflow}
        onWorkflowChange={handleWorkflowChange}
        chunkCount={chunkCount}
      />

      <main className={`main-content ${styles.main}`}>
        <div className={styles.hero}>
          <h1 className={`gradient-text ${styles.heroTitle}`}>
            🏗️ Terraform Architect
          </h1>
          <p className={styles.heroSub}>
            Self-healing agentic RAG — grounding, validation, and human-in-the-loop trust gates
          </p>
        </div>

        {stages.length > 0 && <PipelineVisualizer stages={stages} />}

        <LogTerminal entries={logs} />

        {!hasStarted && (
          <div className={styles.emptyState}>
            <div className={styles.emptyTitle}>Try an example</div>
            <div className={styles.emptyCards}>
              {EXAMPLE_PROMPTS.map((ex) => (
                <button
                  key={ex.label}
                  className={styles.exampleCard}
                  onClick={() => setPrompt(ex.prompt)}
                >
                  <span className={styles.exampleLabel}>{ex.label}</span>
                  <span className={styles.examplePrompt}>{ex.prompt}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {hitlPaused && (
          <>
            <TerraformViewer files={files} />
            {trust && <TrustScoreCard data={trust} />}
            <HitLPanel
              threadId={threadId}
              workflow={workflow}
              prompt={activePrompt}
              files={files}
              onAction={handleHitL}
              loading={hitlLoading}
            />
          </>
        )}

        {(streamingCode && !showResult && !hitlPaused) && (
          <div className="animate-pulse">
            <TerraformViewer files={{ "Generating...": streamingCode + " ▋" }} />
          </div>
        )}

        {showResult && !hitlPaused && (
          <>
            {trust && <TrustScoreCard data={trust} />}
            <TerraformViewer files={files} />
            {(costEstimate || totalRetries > 0) && (
              <div className={styles.runMeta}>
                {costEstimate && (
                  <span className={`badge badge-purple ${styles.metaBadge}`}>💰 {costEstimate}</span>
                )}
                {totalRetries > 0 && (
                  <span className={`badge badge-yellow ${styles.metaBadge}`}>🔧 {totalRetries} self-heal {totalRetries === 1 ? "retry" : "retries"}</span>
                )}
                {totalRetries === 0 && (
                  <span className={`badge badge-green ${styles.metaBadge}`}>✅ Generated first-try</span>
                )}
              </div>
            )}
            {citations.length > 0 && (
              <div className={styles.citations}>
                <div className={styles.citationsLabel}>📚 Sources Used</div>
                <div className={styles.citationsList}>
                  {citations.map((c, i) => (
                    <span key={i} className={`badge badge-blue ${styles.citation}`}>{c}</span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        <div className={styles.inputArea}>
          <div className={styles.inputWrapper}>
            <textarea
              ref={textareaRef}
              className={`input ${styles.textarea}`}
              placeholder="Describe the AWS infrastructure you want to build… e.g. 'Deploy a highly-available 3-tier web app with RDS and CloudFront'"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
              }}
              rows={2}
              disabled={running || hitlPaused}
            />
            <button
              className={`btn btn-primary ${styles.sendBtn}`}
              onClick={handleSubmit}
              disabled={running || hitlPaused || !prompt.trim()}
            >
              <SendHorizonal size={16} />
              {running ? "Running…" : "Generate"}
            </button>
          </div>
          <div className={styles.inputHint}>Press Enter to submit · Shift+Enter for new line</div>
        </div>
      </main>
    </div>
  );
}
