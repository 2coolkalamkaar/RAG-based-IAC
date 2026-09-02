"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import { Settings as SettingsIcon, ShieldCheck, Key, ExternalLink, Loader2, Save, ShieldAlert, CheckCircle2 } from "lucide-react";
import styles from "./page.module.css";
import { getHealth } from "@/lib/api";

export default function SettingsPage() {
  const [roleArn, setRoleArn] = useState("");
  const [extId, setExtId] = useState("");
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [msg, setMsg] = useState({ text: "", type: "" });
  const [testResult, setTestResult] = useState<{success: boolean, message: string} | null>(null);

  useEffect(() => {
    fetch("/api/settings/credentials")
      .then((r) => r.json())
      .then((d) => {
        setRoleArn(d.role_arn || "");
        setExtId(d.external_id || "");
      })
      .catch(() => {});
  }, []);

  async function handleSave() {
    setLoading(true);
    setMsg({ text: "", type: "" });
    try {
      const res = await fetch("/api/settings/credentials", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role_arn: roleArn, external_id: extId }),
      });
      if (res.ok) setMsg({ text: "Credentials saved.", type: "ok" });
      else setMsg({ text: "Failed to save.", type: "err" });
    } catch {
      setMsg({ text: "Network error.", type: "err" });
    } finally {
      setLoading(false);
    }
  }

  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch("/api/settings/credentials/test", { method: "POST" });
      const data = await res.json();
      setTestResult({ success: data.status === "ok", message: data.message });
    } catch {
      setTestResult({ success: false, message: "Network error during test." });
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="app-layout">
      <Sidebar selectedWorkflow="settings" onWorkflowChange={() => {}} />

      <main className={`main-content ${styles.main}`}>
        <div className={styles.header}>
          <div className={styles.titleArea}>
            <h1 className={styles.title}><SettingsIcon size={24} className={styles.titleIcon} /> Settings</h1>
            <p className={styles.subtitle}>Configure TerraForge system parameters and AWS connectivity</p>
          </div>
        </div>

        <div className={styles.contentGrid}>
          {/* STS Credentials Card */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><ShieldCheck size={20} /></div>
              <div>
                <h2 className={styles.cardTitle}>AWS STS Credentials</h2>
                <p className={styles.cardSub}>Configure secure AssumeRole access without static secrets</p>
              </div>
            </div>

            <div className={styles.cardBody}>
              <div className={styles.formGroup}>
                <label className={styles.label}>IAM Role ARN</label>
                <input
                  className="input"
                  placeholder="arn:aws:iam::123456789012:role/TerraForgeDeployer"
                  value={roleArn}
                  onChange={(e) => setRoleArn(e.target.value)}
                />
                <div className={styles.hint}>The role TerraForge will assume to execute Terraform plans.</div>
              </div>

              <div className={styles.formGroup}>
                <label className={styles.label}>External ID (Optional)</label>
                <input
                  className="input"
                  placeholder="e.g. org-123-sec"
                  value={extId}
                  onChange={(e) => setExtId(e.target.value)}
                />
                <div className={styles.hint}>Required if the role's trust policy enforces an ExternalId condition.</div>
              </div>

              <div className={styles.actions}>
                {msg.text && (
                  <span className={`${styles.msg} ${styles[msg.type]}`}>{msg.text}</span>
                )}
                <div className={styles.actionBtns}>
                  <button className="btn btn-ghost" onClick={handleTest} disabled={testing || !roleArn}>
                    {testing ? <Loader2 size={15} className="animate-spin" /> : <Key size={15} />}
                    Test Connection
                  </button>
                  <button className="btn btn-primary" onClick={handleSave} disabled={loading}>
                    {loading ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                    Save Config
                  </button>
                </div>
              </div>

              {testResult && (
                <div className={`${styles.testResult} ${testResult.success ? styles.testOk : styles.testErr}`}>
                  {testResult.success ? <CheckCircle2 size={18} /> : <ShieldAlert size={18} />}
                  <div>
                    <div className={styles.testTitle}>{testResult.success ? "Connection Successful" : "Connection Failed"}</div>
                    <div className={styles.testMsg}>{testResult.message}</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* System Information */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon} style={{ color: "var(--purple)", background: "rgba(168,85,247,0.1)", borderColor: "rgba(168,85,247,0.2)" }}>
                <ExternalLink size={20} />
              </div>
              <div>
                <h2 className={styles.cardTitle}>System Information</h2>
                <p className={styles.cardSub}>Agentic RAG Backend Status</p>
              </div>
            </div>
            <div className={styles.cardBody}>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Architecture</span>
                <span className={styles.infoValue}>LangGraph v0.1</span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>LLM Model</span>
                <span className={styles.infoValue}>Claude 3.5 Sonnet (AWS Bedrock)</span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Vector Store</span>
                <span className={styles.infoValue}>ChromaDB (Local)</span>
              </div>
              <div className={styles.infoRow}>
                <span className={styles.infoLabel}>Circuit Breaker</span>
                <span className={styles.infoValue}>Enabled via SQLite (job_store.db)</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
