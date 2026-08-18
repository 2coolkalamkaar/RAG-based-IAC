"use client";

import styles from "./TrustScoreCard.module.css";

export interface TrustData {
  score: number;
  label: string;
  factors: Record<string, number>;
  explanation: string;
}

const FACTOR_META: Record<string, { label: string; weight: number; icon: string }> = {
  retrieval_similarity: { label: "Retrieval Similarity", weight: 35, icon: "📚" },
  reranker_score_norm:  { label: "Reranker Confidence",  weight: 35, icon: "🎯" },
  validation_passed:    { label: "Validation Pass",       weight: 30, icon: "✅" },
  checkov_passed:       { label: "Security Scan",         weight: 20, icon: "🛡️" },
};

const CHECKOV_WEIGHTS: Record<string, number> = {
  retrieval_similarity: 25,
  reranker_score_norm:  25,
  validation_passed:    30,
  checkov_passed:       20,
};

function getTier(label: string) {
  if (label.includes("High Trust")) return "high";
  if (label.includes("Review"))     return "review";
  return "low";
}

interface Props {
  data: TrustData;
}

export function TrustScoreCard({ data }: Props) {
  const tier = getTier(data.label);
  const scorePct = Math.round(data.score * 100);
  const hasCheckov = "checkov_passed" in data.factors;

  return (
    <div className={`${styles.card} ${styles[tier]}`}>
      <div className={styles.scoreSection}>
        <div className={styles.score}>{scorePct}%</div>
        <div className={styles.scoreLabel}>TRUST SCORE</div>
      </div>

      <div className={styles.details}>
        <div className={styles.label}>{data.label}</div>

        <div className={styles.factors}>
          {Object.entries(FACTOR_META).map(([key, meta]) => {
            if (key === "checkov_passed" && !hasCheckov) return null;
            const val = data.factors[key] ?? 0;
            const pct = Math.round(val * 100);
            const weight = hasCheckov ? CHECKOV_WEIGHTS[key] : meta.weight;
            return (
              <div key={key} className={styles.factor}>
                <div className={styles.factorHeader}>
                  <span className={styles.factorLabel}>
                    {meta.icon} {meta.label}
                    <span className={styles.factorWeight}>(weight {weight}%)</span>
                  </span>
                  <span className={styles.factorPct}>{pct}%</span>
                </div>
                <div className={styles.barTrack}>
                  <div className={styles.barFill} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>

        {data.explanation && (
          <div className={styles.explanation}>
            <div className={styles.explanationTitle}>Why this score?</div>
            <div className={styles.explanationText}>{data.explanation}</div>
          </div>
        )}
      </div>
    </div>
  );
}
