"use client";

import styles from "./PipelineVisualizer.module.css";

export interface Stage {
  id: string;
  icon: string;
  label: string;
  status: "idle" | "running" | "done" | "warning" | "failed" | "paused";
}

const STATUS_LABELS: Record<Stage["status"], string> = {
  idle: "Waiting",
  running: "Running…",
  done: "Complete ✓",
  warning: "Warning ⚠",
  failed: "Failed ✗",
  paused: "Paused ⏸",
};

interface Props {
  stages: Stage[];
}

export function PipelineVisualizer({ stages }: Props) {
  return (
    <div className={styles.wrapper}>
      {stages.map((stage, i) => (
        <div key={stage.id} className={styles.stageWrapper}>
          <div className={`${styles.card} ${styles[stage.status]}`}>
            <div className={styles.icon}>{stage.icon}</div>
            <div className={styles.label}>{stage.label}</div>
            <div className={styles.status}>{STATUS_LABELS[stage.status]}</div>
            {stage.status === "running" && (
              <div className={styles.pulse} />
            )}
          </div>
          {i < stages.length - 1 && (
            <div className={`${styles.connector} ${stage.status === "done" ? styles.connectorActive : ""}`} />
          )}
        </div>
      ))}
    </div>
  );
}
