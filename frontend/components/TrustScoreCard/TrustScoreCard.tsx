"use client";
import styles from "./TrustScoreCard.module.css";
import { ShieldCheck, ShieldAlert, Shield, Info, Activity } from "lucide-react";

export interface TrustData {
  score: number;
  label: string;
  factors: Record<string, number>;
  explanation: string;
}

const FACTOR_META: Record<string, { label: string; weight: number; icon: string }> = {
  retrieval_similarity: { label: "Context grounding", weight: 35, icon: "📚" },
  reranker_score_norm:  { label: "AI confidence",     weight: 35, icon: "🧠" },
  validation_passed:    { label: "Syntax valid",      weight: 30, icon: "✨" },
  checkov_passed:       { label: "Security scan",     weight: 20, icon: "🔒" },
};

const CHECKOV_WEIGHTS: Record<string, number> = {
  retrieval_similarity: 25,
  reranker_score_norm:  25,
  validation_passed:    30,
  checkov_passed:       20,
};

function getTier(label: string) {
  if (label.includes("High Trust")) return "high";
  if (label.includes("Review") || label.includes("Cost Ceiling")) return "review";
  return "low";
}

interface Props { data: TrustData }

export function TrustScoreCard({ data }: Props) {
  const tier = getTier(data.label);
  const scorePct = Math.round(data.score * 100);
  const hasCheckov = "checkov_passed" in data.factors;

  const Icon = tier === "high" ? ShieldCheck : tier === "review" ? Shield : ShieldAlert;

  return (
    <div className={`${styles.card} ${styles[tier]}`}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.iconWrapper}>
            <Icon size={18} />
          </div>
          <div className={styles.headerText}>
            <div className={styles.scoreLabel}>Trust Score</div>
            <div className={styles.label}>{data.label}</div>
          </div>
        </div>
        <div className={styles.scoreCircle}>
          <svg viewBox="0 0 36 36" className={styles.circularChart}>
            <path className={styles.circleBg}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            <path className={styles.circle}
              strokeDasharray={`${scorePct}, 100`}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
            <text x="18" y="21" className={styles.percentage}>{scorePct}%</text>
          </svg>
        </div>
      </div>

      <div className={styles.body}>
        <div className={styles.factorsGrid}>
          {Object.entries(FACTOR_META).map(([key, meta]) => {
            if (key === "checkov_passed" && !hasCheckov) return null;
            const val = data.factors[key] ?? 0;
            const pct = Math.round(val * 100);
            const weight = hasCheckov ? CHECKOV_WEIGHTS[key] : meta.weight;
            return (
              <div key={key} className={styles.factor}>
                <div className={styles.factorTop}>
                  <span className={styles.factorName}>{meta.label}</span>
                  <span className={styles.factorPct}>{pct}%</span>
                </div>
                <div className={styles.barTrack}>
                  <div className={styles.barFill} style={{ width: `${pct}%` }} />
                </div>
                <div className={styles.factorWeight}>Weight: {weight}%</div>
              </div>
            );
          })}
        </div>

        {data.explanation && (
          <div className={styles.explanation}>
            <Activity size={12} className={styles.expIcon} />
            <div className={styles.expText}>{data.explanation}</div>
          </div>
        )}
      </div>
    </div>
  );
}
