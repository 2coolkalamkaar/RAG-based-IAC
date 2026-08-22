"use client";
import styles from "./PipelineVisualizer.module.css";

export interface Stage {
  id: string;
  icon: string;
  label: string;
  status: "idle" | "running" | "done" | "warning" | "failed" | "paused";
}

const STATUS_CONFIG: Record<Stage["status"], { label: string; color: string }> = {
  idle:    { label: "Waiting",    color: "var(--text-dim)" },
  running: { label: "Running",    color: "var(--blue)" },
  done:    { label: "Complete",   color: "var(--green)" },
  warning: { label: "Warning",    color: "var(--yellow)" },
  failed:  { label: "Failed",     color: "var(--red)" },
  paused:  { label: "Paused",     color: "var(--yellow)" },
};

interface Props { stages: Stage[] }

export function PipelineVisualizer({ stages }: Props) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.track}>
        {stages.map((stage, i) => {
          const cfg = STATUS_CONFIG[stage.status];
          const isActive = stage.status === "running";
          const isDone   = stage.status === "done";

          return (
            <div key={stage.id} className={styles.stageRow}>
              <div
                className={`${styles.node} ${styles[`node_${stage.status}`]}`}
                title={`${stage.label}: ${cfg.label}`}
              >
                {isActive && <span className={styles.ring} />}
                <span className={styles.nodeIcon}>{stage.icon}</span>
                <span
                  className={styles.nodeDot}
                  style={{ background: cfg.color, boxShadow: `0 0 6px ${cfg.color}` }}
                />
              </div>
              <div className={styles.nodeLabel} style={{ color: cfg.color }}>
                {stage.label}
              </div>
              <div className={styles.nodeStatus}>{cfg.label}</div>

              {i < stages.length - 1 && (
                <div className={`${styles.connector} ${isDone ? styles.connectorDone : ""}`}>
                  <div className={`${styles.connectorLine} ${isDone ? styles.connectorLineDone : ""}`} />
                  {isDone && <div className={styles.connectorArrow} />}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
