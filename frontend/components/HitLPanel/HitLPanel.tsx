"use client";

import { useState } from "react";
import {
  CheckCircle2, XCircle, Pencil, Rocket, Trash2,
  AlertTriangle, ShieldAlert, DollarSign, ChevronRight,
  Loader2, Terminal
} from "lucide-react";
import styles from "./HitLPanel.module.css";

interface PlanSummary {
  create?: number;
  update?: number;
  delete?: number;
  resources?: { address: string; actions: string[] }[];
}

interface Props {
  threadId: string;
  workflow: string;
  prompt: string;
  files: Record<string, string>;
  planSummary?: PlanSummary;
  costEstimate?: number;
  blastRadiusPassed?: boolean;
  costCeilingPassed?: boolean;
  onAction: (
    action: "approve" | "patch" | "apply" | "destroy",
    patchRequest?: string,
    overrideConfirmed?: boolean,
  ) => Promise<void>;
  loading: boolean;
  applyStatus?: string;
}

export function HitLPanel({
  threadId,
  workflow,
  prompt,
  files,
  planSummary,
  costEstimate,
  blastRadiusPassed = true,
  costCeilingPassed = true,
  onAction,
  loading,
  applyStatus,
}: Props) {
  const [tab, setTab] = useState<"review" | "patch">("review");
  const [patchText, setPatchText] = useState("");
  const [confirmText, setConfirmText] = useState("");
  const [showConfirm, setShowConfirm] = useState(false);
  const [pendingAction, setPendingAction] = useState<"apply" | "destroy" | null>(null);

  const fileCount = Object.keys(files).length;
  const hasGuardFailure = !blastRadiusPassed || !costCeilingPassed;
  const CONFIRM_PHRASE = `confirm-${threadId.slice(0, 6)}`;

  function triggerAction(action: "apply" | "destroy") {
    if (hasGuardFailure && action === "apply") {
      setPendingAction(action);
      setShowConfirm(true);
    } else {
      onAction(action);
    }
  }

  async function handleConfirmedAction() {
    if (confirmText !== CONFIRM_PHRASE) return;
    setShowConfirm(false);
    setConfirmText("");
    await onAction(pendingAction!, "", true);
    setPendingAction(null);
  }

  const applied   = applyStatus === "applied";
  const failed    = applyStatus === "failed";
  const destroyed = applyStatus === "destroyed";

  return (
    <div className={styles.panel}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.pausedBadge}>
            <span className={styles.pausedDot} />
            Awaiting Review
          </div>
          <span className={styles.fileCount}>{fileCount} file{fileCount !== 1 ? "s" : ""} generated</span>
        </div>
        <div className={styles.headerRight}>
          <span className={styles.threadId} title="Thread ID">{threadId.slice(0, 8)}…</span>
        </div>
      </div>

      {/* Plan Preview (Phase 3) */}
      {planSummary && (costEstimate !== undefined) && (
        <div className={styles.planPreview}>
          <div className={styles.planHeader}>
            <Terminal size={13} className={styles.planIcon} />
            <span className={styles.planTitle}>Real Infrastructure Plan</span>
          </div>

          <div className={styles.planStats}>
            {(planSummary.create ?? 0) > 0 && (
              <div className={`${styles.planStat} ${styles.statCreate}`}>
                <span className={styles.statNum}>{planSummary.create}</span>
                <span className={styles.statLabel}>create</span>
              </div>
            )}
            {(planSummary.update ?? 0) > 0 && (
              <div className={`${styles.planStat} ${styles.statUpdate}`}>
                <span className={styles.statNum}>{planSummary.update}</span>
                <span className={styles.statLabel}>update</span>
              </div>
            )}
            {(planSummary.delete ?? 0) > 0 && (
              <div className={`${styles.planStat} ${styles.statDelete}`}>
                <span className={styles.statNum}>{planSummary.delete}</span>
                <span className={styles.statLabel}>destroy</span>
              </div>
            )}
            {costEstimate !== undefined && costEstimate > 0 && (
              <div className={`${styles.planStat} ${styles.statCost}`}>
                <DollarSign size={11} />
                <span className={styles.statNum}>${costEstimate.toFixed(0)}</span>
                <span className={styles.statLabel}>/mo est.</span>
              </div>
            )}
          </div>

          {/* Guard warnings */}
          {!blastRadiusPassed && (
            <div className={styles.guardAlert}>
              <ShieldAlert size={13} />
              <span>Blast-radius guard flagged this plan — it touches resources outside this job. Override required to apply.</span>
            </div>
          )}
          {!costCeilingPassed && (
            <div className={`${styles.guardAlert} ${styles.guardCost}`}>
              <AlertTriangle size={13} />
              <span>Estimated cost exceeds $100/mo ceiling. Override required to apply.</span>
            </div>
          )}
        </div>
      )}

      {/* Apply status banner */}
      {applied && (
        <div className={`${styles.statusBanner} ${styles.statusApplied}`}>
          <CheckCircle2 size={16} />
          <span>Successfully applied to AWS</span>
        </div>
      )}
      {failed && (
        <div className={`${styles.statusBanner} ${styles.statusFailed}`}>
          <XCircle size={16} />
          <span>Apply failed — check backend logs</span>
        </div>
      )}
      {destroyed && (
        <div className={`${styles.statusBanner} ${styles.statusDestroyed}`}>
          <Trash2 size={16} />
          <span>Resources destroyed from AWS</span>
        </div>
      )}

      {/* Tabs */}
      {!applied && !destroyed && (
        <>
          <div className={styles.tabs}>
            <button
              className={`${styles.tab} ${tab === "review" ? styles.tabActive : ""}`}
              onClick={() => setTab("review")}
            >
              Review &amp; Action
            </button>
            <button
              className={`${styles.tab} ${tab === "patch" ? styles.tabActive : ""}`}
              onClick={() => setTab("patch")}
            >
              Request Changes
            </button>
          </div>

          {tab === "review" && (
            <div className={styles.actionGrid}>
              {/* Approve (save only) */}
              <button
                className={`btn btn-success ${styles.actionBtn}`}
                onClick={() => onAction("approve")}
                disabled={loading}
              >
                {loading ? <span className="spinner" /> : <CheckCircle2 size={15} />}
                Approve &amp; Save
              </button>

              {/* Apply to AWS */}
              {workflow === "hitl" && (
                <button
                  className={`btn btn-apply ${styles.actionBtn}`}
                  onClick={() => triggerAction("apply")}
                  disabled={loading || applied}
                >
                  {loading ? <span className="spinner" /> : <Rocket size={15} />}
                  {hasGuardFailure ? "⚠ Apply to AWS" : "Apply to AWS"}
                </button>
              )}

              {/* Reject */}
              <button
                className={`btn btn-danger ${styles.actionBtn}`}
                onClick={() => onAction("patch", "Start fresh — reject current code.")}
                disabled={loading}
              >
                <XCircle size={15} />
                Reject
              </button>
            </div>
          )}

          {tab === "patch" && (
            <div className={styles.patchArea}>
              <div className={styles.patchLabel}>
                Describe the change you want — the agent will apply it surgically
              </div>
              <textarea
                className={`input ${styles.patchInput}`}
                placeholder="e.g. Add CloudWatch logging to the Lambda, use t3.small instead of t3.micro…"
                value={patchText}
                onChange={(e) => setPatchText(e.target.value)}
                rows={3}
              />
              <button
                className={`btn btn-primary ${styles.patchBtn}`}
                onClick={() => { onAction("patch", patchText); setPatchText(""); }}
                disabled={loading || !patchText.trim()}
              >
                {loading ? <span className="spinner" /> : <Pencil size={14} />}
                Apply Patch
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </>
      )}

      {/* Typed confirmation modal */}
      {showConfirm && (
        <div className={styles.confirmOverlay}>
          <div className={styles.confirmBox}>
            <div className={styles.confirmIcon}>
              <ShieldAlert size={22} style={{ color: "var(--yellow)" }} />
            </div>
            <div className={styles.confirmTitle}>Override Required</div>
            <div className={styles.confirmBody}>
              {!blastRadiusPassed && (
                <p>This plan touches resources <strong>not created by this job</strong>. Applying could affect existing infrastructure.</p>
              )}
              {!costCeilingPassed && (
                <p>Estimated cost exceeds the $100/mo ceiling.</p>
              )}
              <p>Type <code className={styles.confirmCode}>{CONFIRM_PHRASE}</code> to proceed:</p>
            </div>
            <input
              className={`input ${styles.confirmInput}`}
              placeholder={CONFIRM_PHRASE}
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              autoFocus
            />
            <div className={styles.confirmActions}>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => { setShowConfirm(false); setConfirmText(""); }}
              >
                Cancel
              </button>
              <button
                className="btn btn-danger btn-sm"
                disabled={confirmText !== CONFIRM_PHRASE || loading}
                onClick={handleConfirmedAction}
              >
                {loading ? <Loader2 size={13} className="animate-spin" /> : <ShieldAlert size={13} />}
                Override &amp; Apply
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
