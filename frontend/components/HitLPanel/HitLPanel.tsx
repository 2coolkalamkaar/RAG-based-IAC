"use client";

import { useState } from "react";
import { CheckCircle, MessageSquare } from "lucide-react";
import styles from "./HitLPanel.module.css";

interface Props {
  threadId: string;
  workflow: string;
  prompt: string;
  files: Record<string, string>;
  onAction: (action: "approve" | "patch", patchRequest?: string) => void;
  loading?: boolean;
}

export function HitLPanel({ files, onAction, loading }: Props) {
  const [patchRequest, setPatchRequest] = useState("");
  const [mode, setMode] = useState<"idle" | "patch">("idle");

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.indicator} />
        <div>
          <div className={styles.title}>⏸ Awaiting Human Review</div>
          <div className={styles.subtitle}>
            Review the generated Terraform code above and choose an action.
          </div>
        </div>
      </div>

      <div className={styles.fileCount}>
        {Object.keys(files).length} file(s) ready for review
      </div>

      <div className={styles.actions}>
        <button
          className={`btn btn-success ${styles.actionBtn}`}
          onClick={() => onAction("approve")}
          disabled={loading}
        >
          <CheckCircle size={16} />
          Approve & Finalize
        </button>

        <button
          className={`btn btn-ghost ${styles.actionBtn}`}
          onClick={() => setMode(mode === "patch" ? "idle" : "patch")}
          disabled={loading}
        >
          <MessageSquare size={16} />
          Request Changes
        </button>
      </div>

      {mode === "patch" && (
        <div className={styles.patchForm}>
          <textarea
            className="input"
            placeholder="Describe what you'd like to change… e.g. 'Add a read replica for the RDS instance'"
            value={patchRequest}
            onChange={(e) => setPatchRequest(e.target.value)}
            rows={3}
          />
          <button
            className={`btn btn-primary ${styles.actionBtn}`}
            onClick={() => {
              if (patchRequest.trim()) onAction("patch", patchRequest.trim());
            }}
            disabled={loading || !patchRequest.trim()}
          >
            Apply Patch
          </button>
        </div>
      )}
    </div>
  );
}
