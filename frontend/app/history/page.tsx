"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { getJobs, getJob, deleteJob, Job, JobDetail } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import { TrustScoreCard } from "@/components/TrustScoreCard/TrustScoreCard";
import { TerraformViewer } from "@/components/TerraformViewer/TerraformViewer";
import { History, Trash2, ChevronLeft, RefreshCw } from "lucide-react";
import styles from "./page.module.css";

function trustBadge(score: number | null) {
  if (score === null) return { cls: "badge-purple", label: "N/A" };
  if (score >= 0.75) return { cls: "badge-green", label: `${Math.round(score * 100)}% High` };
  if (score >= 0.50) return { cls: "badge-yellow", label: `${Math.round(score * 100)}% Review` };
  return { cls: "badge-red", label: `${Math.round(score * 100)}% Low` };
}

export default function HistoryPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<JobDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingJob, setLoadingJob] = useState(false);

  async function loadJobs() {
    setLoading(true);
    try {
      setJobs(await getJobs());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadJobs(); }, []);

  async function handleSelect(id: string) {
    setLoadingJob(true);
    try {
      setSelected(await getJob(id));
    } finally {
      setLoadingJob(false);
    }
  }

  async function handleDelete(id: string) {
    await deleteJob(id);
    if (selected?.id === id) setSelected(null);
    loadJobs();
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
      <Sidebar selectedWorkflow="hitl" onWorkflowChange={() => {}} />

      <main className={`main-content ${styles.main}`}>
        <div className={styles.header}>
          <Link href="/" className={`btn btn-ghost ${styles.backBtn}`}>
            <ChevronLeft size={16} /> Back
          </Link>
          <div>
            <h1 className={styles.title}><History size={22} /> Job History</h1>
            <p className={styles.subtitle}>Browse and inspect all approved Terraform runs</p>
          </div>
          <button className={`btn btn-ghost`} onClick={loadJobs} title="Refresh">
            <RefreshCw size={15} />
          </button>
        </div>

        <div className={styles.layout}>
          {/* Job List */}
          <div className={styles.list}>
            {loading && <div className={styles.empty}>Loading…</div>}
            {!loading && jobs.length === 0 && (
              <div className={styles.empty}>
                No jobs yet. Approve a HitL run to save your first job!
              </div>
            )}
            {jobs.map((job) => {
              const badge = trustBadge(job.trust_score);
              const wfCls = WORKFLOW_COLORS[job.workflow] ?? "badge-purple";
              return (
                <div
                  key={job.id}
                  className={`${styles.jobCard} ${selected?.id === job.id ? styles.jobCardActive : ""}`}
                  onClick={() => handleSelect(job.id)}
                >
                  <div className={styles.jobTop}>
                    <span className={`badge ${wfCls}`}>{job.workflow}</span>
                    <span className={`badge ${badge.cls}`}>{badge.label}</span>
                    <button
                      className={styles.deleteBtn}
                      onClick={(e) => { e.stopPropagation(); handleDelete(job.id); }}
                      title="Delete"
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
            {loadingJob && <div className={styles.empty}>Loading job…</div>}
            {!loadingJob && !selected && (
              <div className={styles.empty}>
                ← Select a job from the list to view its Terraform files and trust score
              </div>
            )}
            {!loadingJob && selected && (
              <>
                <div className={styles.detailHeader}>
                  <h2 className={styles.detailTitle}>{selected.prompt}</h2>
                  <div className={styles.detailMeta}>
                    <span className={`badge ${WORKFLOW_COLORS[selected.workflow] ?? "badge-purple"}`}>
                      {selected.workflow}
                    </span>
                    <span className={styles.detailDate}>{formatDate(selected.created_at)}</span>
                  </div>
                </div>

                {selected.trust_score !== null && selected.trust_label && (
                  <TrustScoreCard data={{
                    score: selected.trust_score,
                    label: selected.trust_label,
                    factors: {},
                    explanation: "",
                  }} />
                )}

                <TerraformViewer files={selected.files} />
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
