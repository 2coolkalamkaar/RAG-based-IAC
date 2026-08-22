"use client";

import { useState } from "react";
import { Download, FolderDown } from "lucide-react";
import styles from "./TerraformViewer.module.css";

interface Props {
  files: Record<string, string>;
}

export function TerraformViewer({ files }: Props) {
  const fileNames = Object.keys(files);
  const [active, setActive] = useState(fileNames[0] ?? "");

  if (fileNames.length === 0) return null;

  const activeCode = files[active] ?? "";

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
    // Build a simple text bundle with file separators so it's one download
    const bundle = fileNames
      .map((n) => `# ===== ${n} =====\n\n${files[n]}`)
      .join("\n\n");
    const blob = new Blob([bundle], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "terraform-output.tf";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.tabBar}>
        <div className={styles.tabs}>
          {fileNames.map((name) => (
            <button
              key={name}
              className={`${styles.tab} ${name === active ? styles.tabActive : ""}`}
              onClick={() => setActive(name)}
            >
              📄 {name}
            </button>
          ))}
        </div>
        <div className={styles.actions}>
          <button
            className={styles.downloadBtn}
            onClick={() => downloadFile(active, activeCode)}
            title={`Download ${active}`}
          >
            <Download size={13} /> {active}
          </button>
          {fileNames.length > 1 && (
            <button
              className={`${styles.downloadBtn} ${styles.downloadAllBtn}`}
              onClick={downloadAll}
              title="Download all files bundled"
            >
              <FolderDown size={13} /> Download All ({fileNames.length} files)
            </button>
          )}
        </div>
      </div>
      <div className={styles.codeWrapper}>
        <pre className={styles.code}>
          <code>{activeCode}</code>
        </pre>
      </div>
    </div>
  );
}
