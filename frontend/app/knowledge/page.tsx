"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { uploadDoc, listDocs, InternalDoc } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar/Sidebar";
import { BookOpen, UploadCloud, ChevronLeft, CheckCircle, AlertCircle } from "lucide-react";
import styles from "./page.module.css";

const SUPPORTED = [".md", ".txt", ".tf", ".hcl", ".pdf"];

export default function KnowledgePage() {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<{ chunks: number; name: string } | null>(null);
  const [error, setError] = useState("");
  const [docs, setDocs] = useState<InternalDoc[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);

  async function loadDocs() {
    setLoadingDocs(true);
    try { setDocs(await listDocs()); } finally { setLoadingDocs(false); }
  }
  useEffect(() => { loadDocs(); }, []);

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
    setUploading(true);
    setError("");
    setResult(null);
    try {
      const res = await uploadDoc(file, description);
      setResult({ chunks: res.chunks_added, name: res.filename });
      setFile(null);
      setDescription("");
      loadDocs();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function formatDate(iso: string) {
    return new Date(iso).toLocaleString();
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
              Inject proprietary or internal documentation directly into the RAG knowledge base
            </p>
          </div>
        </div>

        <div className={styles.grid}>
          {/* Upload Panel */}
          <div className="card">
            <h2 className={styles.cardTitle}>Upload Document</h2>
            <p className={styles.cardDesc}>
              Supports <code>.md</code>, <code>.tf</code>, <code>.txt</code>, <code>.hcl</code>, <code>.pdf</code>
            </p>

            {/* Drop Zone */}
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
                  <div className={styles.dropLabel}>Drag & drop a file, or click to browse</div>
                  <div className={styles.dropSub}>Injected docs are tagged as internal and never overwrite AWS provider docs</div>
                </>
              )}
            </div>

            {/* Description */}
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
            <h2 className={styles.cardTitle}>Injected Documents</h2>
            <p className={styles.cardDesc}>
              These internal docs are active in the RAG pipeline alongside the 1,696 AWS provider docs.
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
