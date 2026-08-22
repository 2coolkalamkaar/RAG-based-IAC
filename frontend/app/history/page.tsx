"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { getJobs, getJob, deleteJob, submitHitLAction, Job, JobDetail } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import { TrustScoreCard } from "@/components/TrustScoreCard/TrustScoreCard";
import { TerraformViewer } from "@/components/TerraformViewer/TerraformViewer";
import { History, Trash2, ChevronLeft, RefreshCw, Rocket, ShieldAlert, Loader2, Info } from "lucide-react";
import styles from "./page.module.css";

function trustBadge(score: number | null) {
  if (score === null) return { cls: "badge-purple", label: "N/A" };
  if (score >= 0.75) return { cls: "badge-green", label: `${Math.round(score * 100)}% High` };
  if (score >= 0.50) return { cls: "badge-yellow", label: `${Math.round(score * 100)}% Review` };
  return { cls: "badge-red", label: `${Math.round(score * 100)}% Low` };
}

function statusBadge(status: string) {
  if (status === "applied") return { cls: "badge-green", label: "Applied" };
  if (status === "failed") return { cls: "badge-red", label: "Apply Failed" };
  if (status === "destroyed") return { cls: "badge-purple", label: "Destroyed" };
  return { cls: "badge-blue", label: "Saved" };
}

export default function HistoryPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<JobDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingJob, setLoadingJob] = useState(false);
  const [destroying, setDestroying] = useState(false);

  async function loadJobs() {
    setLoading(true);
    try { setJobs(await getJobs()); } finally { setLoading(false); }
  }

  useEffect(() => { loadJobs(); }, []);

  async function handleSelect(id: string) {
    setLoadingJob(true);
    try { setSelected(await getJob(id)); } finally { setLoadingJob(false); }
  }

  async function handleDelete(id: string) {
    if (!confirm("Remove this job from history? This does not affect AWS.")) return;
    await deleteJob(id);
    if (selected?.id === id) setSelected(null);
    loadJobs();
  }

  async function handleDestroy() {
    if (!selected || selected.apply_status !== "applied") return;
    if (!confirm("WARNING: This will run `terraform destroy` on AWS and permanently delete the infrastructure. Proceed?")) return;

    setDestroying(true);
    try {
      const res = await submitHitLAction(selected.id, selected.workflow, "destroy");
      const status = (res as any)?.apply_status;
      setSelected({ ...selected, apply_status: status });
      setJobs(jobs.map(j => j.id === selected.id ? { ...j, apply_status: status } : j));
    } catch (e) {
      alert("Destroy failed. Check backend logs.");
    } finally {
      setDestroying(false);
    }
  }

  function formatDate(iso: string) {
    return new Date(iso).toLocaleString();
  }

  const WORKFLOW_COLORS: Record<string, string> = {
    "hitl": "badge-blue", "advanced": "badge-purple", "secure": "badge-green",
    "rag": "badge-blue", "basic": "badge-purple",
  };

  return (
    <div className="app-layout">
      <Sidebar selectedWorkflow="history" onWorkflowChange={() => {}} />

      <main className={`main-content ${styles.main}`}>
        <div className={styles.header}>
          <Link href="/" className={`btn btn-ghost ${styles.backBtn}`}>
            <ChevronLeft size={16} /> Dashboard
          </Link>
          <div className={styles.titleArea}>
            <h1 className={styles.title}><History size={20} className={styles.titleIcon} /> Job History</h1>
            <p className={styles.subtitle}>Browse approved Terraform runs and manage active infrastructure</p>
          </div>
          <button className={`btn btn-ghost ${styles.refreshBtn}`} onClick={loadJobs} title="Refresh">
            <RefreshCw size={15} />
          </button>
        </div>

        <div className={styles.layout}>
          {/* Job List */}
          <div className={styles.list}>
            {loading && <div className={styles.empty}><Loader2 className="animate-spin" /> Loading…</div>}
            {!loading && jobs.length === 0 && (
              <div className={styles.empty}>
                <Info size={24} style={{ opacity: 0.5, marginBottom: '0.5rem' }} />
                No jobs yet. Approve a HitL run to save your first job!
              </div>
            )}
            {jobs.map((job) => {
              const tb = trustBadge(job.trust_score);
              const sb = statusBadge(job.apply_status);
              const wfCls = WORKFLOW_COLORS[job.workflow] ?? "badge-purple";
              return (
                <div
                  key={job.id}
                  className={`${styles.jobCard} ${selected?.id === job.id ? styles.jobCardActive : ""}`}
                  onClick={() => handleSelect(job.id)}
                >
                  <div className={styles.jobTop}>
                    <span className={`badge ${wfCls}`}>{job.workflow}</span>
                    <span className={`badge ${tb.cls}`}>{tb.label}</span>
                    <span className={`badge ${sb.cls}`}>{sb.label}</span>
                    <button
                      className={styles.deleteBtn}
                      onClick={(e) => { e.stopPropagation(); handleDelete(job.id); }}
                      title="Remove from history"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                  <div className={styles.jobPrompt}>{job.prompt}</div>
                  <div className={styles.jobDate}>{formatDate(job.created_at)}</div>
                </div>
              );
            })}
          </div>

          {/* Detail Panel */}
          <div className={styles.detail}>
            {loadingJob && <div className={styles.empty}><Loader2 className="animate-spin" /> Loading job…</div>}
            {!loadingJob && !selected && (
              <div className={styles.emptyState}>
                <div className={styles.emptyStateIcon}><History size={32} /></div>
                <div>Select a job from the list to view its configuration and trust score.</div>
              </div>
            )}
            {!loadingJob && selected && (
              <div className={styles.detailContent}>
                <div className={styles.detailHeader}>
                  <div className={styles.detailHeaderLeft}>
                    <h2 className={styles.detailTitle}>{selected.prompt}</h2>
                    <div className={styles.detailMeta}>
                      <span className={`badge ${WORKFLOW_COLORS[selected.workflow] ?? "badge-purple"}`}>
                        {selected.workflow}
                      </span>
                      <span className={styles.detailDate}>{formatDate(selected.created_at)}</span>
                      <span className={styles.detailId}>ID: {selected.id.slice(0,8)}</span>
                    </div>
                  </div>

                  {/* Destroy action — available for applied jobs */}
                  {selected.apply_status === "applied" && (
                    <div className={styles.actionPanel}>
                      <div className={styles.actionWarning}>⚡ Active on AWS</div>
                      <button
                        className="btn btn-danger"
                        onClick={handleDestroy}
                        disabled={destroying}
                      >
                        {destroying ? <Loader2 size={14} className="animate-spin" /> : <ShieldAlert size={14} />}
                        {destroying ? "Destroying…" : "Destroy Infrastructure"}
                      </button>
                    </div>
                  )}
                  {/* For saved (not yet applied) jobs — also allow triggering destroy */}
                  {(!selected.apply_status || selected.apply_status === "") && (
                    <div className={styles.actionPanel}>
                      <div className={styles.actionWarning} style={{ color: "var(--text-dim)" }}>Saved — not applied to AWS</div>
                    </div>
                  )}
                  {selected.apply_status === "destroyed" && (
                    <div className={styles.actionPanel}>
                      <div className={styles.actionSuccess}>✅ Infrastructure Destroyed</div>
                    </div>
                  )}
                  {selected.apply_status === "failed" && (
                    <div className={styles.actionPanel}>
                      <div className={styles.actionWarning} style={{ color: "var(--red)" }}>⚠️ Apply Failed</div>
                    </div>
                  )}
                </div>

                {selected.trust_score !== null && selected.trust_label && (
                  <TrustScoreCard data={{
                    score: selected.trust_score,
                    label: selected.trust_label,
                    factors: selected.trust_factors ?? {},
                    explanation: selected.trust_explanation ?? "Trust score captured at time of generation.",
                  }} />
                )}

                <TerraformViewer files={selected.files} />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
