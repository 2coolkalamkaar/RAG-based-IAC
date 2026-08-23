"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { uploadDoc, listDocs, InternalDoc } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import {
  BookOpen, UploadCloud, ChevronLeft, CheckCircle, AlertCircle,
  RefreshCw, Database, Zap, Clock, FileText, Play, Loader2
} from "lucide-react";
import styles from "./page.module.css";

const SUPPORTED = [".md", ".txt", ".tf", ".hcl", ".pdf"];
const API = process.env.NEXT_PUBLIC_API_URL || "";

interface ETLManifest {
  provider_version: string;
  last_run: string;
  total_indexed: number;
}

export default function KnowledgePage() {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<{ chunks: number; name: string } | null>(null);
  const [error, setError] = useState("");
  const [docs, setDocs] = useState<InternalDoc[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);

  // ETL state
  const [etlManifest, setEtlManifest] = useState<ETLManifest | null>(null);
  const [etlStatus, setEtlStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [etlLogs, setEtlLogs] = useState<string[]>([]);
  const [fullRebuild, setFullRebuild] = useState(false);
  const [skipPull, setSkipPull] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  async function loadDocs() {
    setLoadingDocs(true);
    try { setDocs(await listDocs()); } finally { setLoadingDocs(false); }
  }

  async function loadEtlStatus() {
    try {
      const res = await fetch(`${API}/api/etl/status`);
      const data = await res.json();
      if (data.manifest) setEtlManifest(data.manifest);
    } catch { /* ignore */ }
  }

  useEffect(() => {
    loadDocs();
    loadEtlStatus();
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [etlLogs]);

  const onDrop = useCallback((f: File) => {
    setError("");
    setResult(null);
    const ext = "." + f.name.split(".").pop()?.toLowerCase();
    if (!SUPPORTED.includes(ext)) {
      setError(`Unsupported file type: ${ext}. Supported: ${SUPPORTED.join(", ")}`);
      return;
    }
    setFile(f);
  }, []);

  function handleDragOver(e: React.DragEvent) { e.preventDefault(); setDragging(true); }
  function handleDragLeave() { setDragging(false); }
  function handleDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onDrop(f);
  }

  async function handleUpload() {
    if (!file) return;
    setUploading(true); setError(""); setResult(null);
    try {
      const res = await uploadDoc(file, description);
      setResult({ chunks: res.chunks_added, name: res.filename });
      setFile(null); setDescription("");
      loadDocs();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally { setUploading(false); }
  }

  async function handleRunETL() {
    setEtlStatus("running");
    setEtlLogs(["🚀 Starting ETL pipeline..."]);

    try {
      const res = await fetch(`${API}/api/etl/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_rebuild: fullRebuild, skip_pull: skipPull }),
      });

      if (!res.ok || !res.body) {
        setEtlLogs(prev => [...prev, `[ERROR] Server returned ${res.status}`]);
        setEtlStatus("error");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("event: done")) {
            setEtlStatus("done");
            await loadEtlStatus();
            await loadDocs();
          } else if (line.startsWith("data: ") && !line.includes("{}")) {
            const raw = line.slice(6);
            try {
              const parsed = JSON.parse(raw);
              setEtlLogs(prev => [...prev, parsed]);
            } catch {
              setEtlLogs(prev => [...prev, raw]);
            }
          }
        }
      }
    } catch (err: unknown) {
      setEtlLogs(prev => [...prev, `[ERROR] ${err instanceof Error ? err.message : String(err)}`]);
      setEtlStatus("error");
    }
  }

  function formatDate(iso: string) {
    return new Date(iso).toLocaleString();
  }

  function getLogLineClass(line: string) {
    if (line.startsWith("[ERROR]") || line.includes("Error")) return styles.logError;
    if (line.includes("✅") || line.includes("Done") || line.includes("Complete")) return styles.logSuccess;
    if (line.includes("⚠️")) return styles.logWarn;
    if (line.includes("📥") || line.includes("Indexing") || line.includes("Batch")) return styles.logInfo;
    return styles.logDefault;
  }

  return (
    <div className="app-layout">
      <Sidebar selectedWorkflow="hitl" onWorkflowChange={() => {}} />

      <main className={`main-content ${styles.main}`}>
        <div className={styles.header}>
          <Link href="/" className={`btn btn-ghost`}>
            <ChevronLeft size={16} /> Back
          </Link>
          <div>
            <h1 className={styles.title}><BookOpen size={22} /> Knowledge Base</h1>
            <p className={styles.subtitle}>
              Manage the RAG vector database — sync provider docs or inject internal documents
            </p>
          </div>
        </div>

        {/* ── ETL Sync Panel ── */}
        <div className={`card ${styles.etlCard}`}>
          <div className={styles.etlHeader}>
            <div className={styles.etlHeaderLeft}>
              <Database size={20} className={styles.etlIcon} />
              <div>
                <h2 className={styles.cardTitle} style={{ margin: 0 }}>Sync Vector Database</h2>
                <p className={styles.cardDesc} style={{ margin: 0 }}>
                  Pull latest AWS Terraform provider docs and update ChromaDB incrementally
                </p>
              </div>
            </div>

            {etlManifest && (
              <div className={styles.etlMeta}>
                <div className={styles.etlMetaItem}>
                  <FileText size={12} />
                  <span>{etlManifest.total_indexed} docs indexed</span>
                </div>
                <div className={styles.etlMetaItem}>
                  <Clock size={12} />
                  <span>Last sync: {formatDate(etlManifest.last_run)}</span>
                </div>
                <div className={styles.etlMetaItem}>
                  <Zap size={12} />
                  <span>Provider: {etlManifest.provider_version}</span>
                </div>
              </div>
            )}
          </div>

          <div className={styles.etlOptions}>
            <label className={styles.etlToggle}>
              <input
                type="checkbox"
                checked={skipPull}
                onChange={e => setSkipPull(e.target.checked)}
                disabled={etlStatus === "running"}
              />
              <span>Skip git pull <small>(use local docs)</small></span>
            </label>
            <label className={`${styles.etlToggle} ${styles.etlToggleDanger}`}>
              <input
                type="checkbox"
                checked={fullRebuild}
                onChange={e => setFullRebuild(e.target.checked)}
                disabled={etlStatus === "running"}
              />
              <span>Full rebuild <small>(wipe &amp; reindex everything)</small></span>
            </label>
            <button
              className={`btn ${etlStatus === "running" ? "btn-ghost" : "btn-primary"} ${styles.etlRunBtn}`}
              onClick={handleRunETL}
              disabled={etlStatus === "running"}
              id="etl-sync-btn"
            >
              {etlStatus === "running" ? (
                <><Loader2 size={15} className="animate-spin" /> Running…</>
              ) : (
                <><Play size={15} /> {fullRebuild ? "Full Rebuild" : "Incremental Sync"}</>
              )}
            </button>
          </div>

          {etlLogs.length > 0 && (
            <div className={styles.etlTerminal}>
              <div className={styles.etlTerminalBar}>
                <span className={styles.termDot} style={{ background: "#ff5f57" }} />
                <span className={styles.termDot} style={{ background: "#febc2e" }} />
                <span className={styles.termDot} style={{ background: "#28c840" }} />
                <span className={styles.termTitle}>
                  {etlStatus === "running" && <Loader2 size={11} className="animate-spin" />}
                  {etlStatus === "done" && <CheckCircle size={11} style={{ color: "var(--green)" }} />}
                  {etlStatus === "error" && <AlertCircle size={11} style={{ color: "var(--red)" }} />}
                  &nbsp;ETL Pipeline Log
                </span>
                <button
                  className={styles.termClear}
                  onClick={() => { setEtlLogs([]); setEtlStatus("idle"); }}
                >Clear</button>
              </div>
              <div className={styles.etlLogs}>
                {etlLogs.map((line, i) => (
                  <div key={i} className={`${styles.logLine} ${getLogLineClass(line)}`}>
                    <span className={styles.logNum}>{i + 1}</span>
                    <span className={styles.logText}>{line}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
              {etlStatus === "done" && (
                <div className={styles.etlDoneBanner}>
                  <CheckCircle size={14} /> Vector DB sync complete — pipeline will use updated docs on next request
                </div>
              )}
            </div>
          )}
        </div>

        <div className={styles.grid}>
          {/* Upload Panel */}
          <div className="card">
            <h2 className={styles.cardTitle}>Upload Internal Document</h2>
            <p className={styles.cardDesc}>
              Supports <code>.md</code>, <code>.tf</code>, <code>.txt</code>, <code>.hcl</code>, <code>.pdf</code>
            </p>

            <div
              className={`${styles.dropzone} ${dragging ? styles.dragOver : ""} ${file ? styles.hasFile : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById("fileInput")?.click()}
            >
              <input
                id="fileInput"
                type="file"
                className={styles.fileInput}
                accept={SUPPORTED.join(",")}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) onDrop(f); }}
              />
              {file ? (
                <>
                  <div className={styles.dropIcon}>📄</div>
                  <div className={styles.dropFileName}>{file.name}</div>
                  <div className={styles.dropFileSize}>{(file.size / 1024).toFixed(1)} KB</div>
                </>
              ) : (
                <>
                  <UploadCloud size={32} className={styles.dropIcon} />
                  <div className={styles.dropLabel}>Drag &amp; drop a file, or click to browse</div>
                  <div className={styles.dropSub}>Injected docs are tagged as internal and never overwrite AWS provider docs</div>
                </>
              )}
            </div>

            <div className={styles.field}>
              <label className={styles.label}>Description (optional)</label>
              <input
                className="input"
                placeholder="e.g. 'Internal ACME Corp VPC module schema'"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>

            {error && (
              <div className={styles.errorBanner}>
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            {result && (
              <div className={styles.successBanner}>
                <CheckCircle size={16} />
                Injected <strong>{result.name}</strong> — {result.chunks} chunks added to knowledge base
              </div>
            )}

            <button
              className={`btn btn-primary btn-full ${styles.uploadBtn}`}
              onClick={handleUpload}
              disabled={!file || uploading}
            >
              <UploadCloud size={16} />
              {uploading ? "Injecting…" : "Inject into Knowledge Base"}
            </button>
          </div>

          {/* Injected Docs List */}
          <div className="card">
            <div className={styles.docsHeader}>
              <h2 className={styles.cardTitle}>Injected Documents</h2>
              <button className="btn btn-ghost" onClick={loadDocs} title="Refresh">
                <RefreshCw size={14} />
              </button>
            </div>
            <p className={styles.cardDesc}>
              These internal docs are active in the RAG pipeline alongside the AWS provider docs.
            </p>

            {loadingDocs && <div className={styles.emptyDocs}>Loading…</div>}
            {!loadingDocs && docs.length === 0 && (
              <div className={styles.emptyDocs}>
                No internal docs injected yet. Upload your first document to get started!
              </div>
            )}
            <div className={styles.docList}>
              {docs.map((doc) => (
                <div key={doc.filename} className={styles.docItem}>
                  <div className={styles.docIcon}>📄</div>
                  <div className={styles.docInfo}>
                    <div className={styles.docName}>{doc.filename}</div>
                    {doc.description && (
                      <div className={styles.docDesc}>{doc.description}</div>
                    )}
                    <div className={styles.docDate}>{formatDate(doc.injected_at)}</div>
                  </div>
                  <span className="badge badge-purple" style={{ fontSize: "0.65rem" }}>internal</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
