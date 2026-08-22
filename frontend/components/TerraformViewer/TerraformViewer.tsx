"use client";

import { useState } from "react";
import { Download, FolderDown, FileCode2, Copy, Check } from "lucide-react";
import styles from "./TerraformViewer.module.css";

interface Props {
  files: Record<string, string>;
}

export function TerraformViewer({ files }: Props) {
  const fileNames = Object.keys(files);
  const [active, setActive] = useState(fileNames[0] ?? "");
  const [copied, setCopied] = useState(false);

  if (fileNames.length === 0) return null;

  const activeCode = files[active] ?? "";

  function handleCopy() {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function downloadFile(name: string, code: string) {
    const blob = new Blob([code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadAll() {
    const bundle = fileNames
      .map((n) => `# ===== ${n} =====\n\n${files[n]}`)
      .join("\n\n");
    const blob = new Blob([bundle], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "terraform-bundle.tf";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={styles.wrapper}>
      {/* Header bar */}
      <div className={styles.header}>
        <div className={styles.tabs}>
          {fileNames.map((name) => (
            <button
              key={name}
              className={`${styles.tab} ${name === active ? styles.tabActive : ""}`}
              onClick={() => setActive(name)}
            >
              <FileCode2 size={14} className={styles.tabIcon} />
              {name}
            </button>
          ))}
        </div>
        <div className={styles.actions}>
          <button className={styles.actionBtn} onClick={handleCopy} title="Copy code">
            {copied ? <Check size={14} className={styles.copiedIcon} /> : <Copy size={14} />}
          </button>
          <button className={styles.actionBtn} onClick={() => downloadFile(active, activeCode)} title="Download current file">
            <Download size={14} />
          </button>
          {fileNames.length > 1 && (
            <button className={`${styles.actionBtn} ${styles.downloadAllBtn}`} onClick={downloadAll} title="Download all files">
              <FolderDown size={14} /> <span>Download All</span>
            </button>
          )}
        </div>
      </div>

      {/* Code Area */}
      <div className={styles.codeContainer}>
        {/* Fake line numbers for aesthetics */}
        <div className={styles.lineNumbers} aria-hidden="true">
          {activeCode.split('\n').map((_, i) => (
            <div key={i} className={styles.lineNumber}>{i + 1}</div>
          ))}
        </div>
        <pre className={styles.code}>
          <code>{activeCode}</code>
        </pre>
      </div>
    </div>
  );
}
