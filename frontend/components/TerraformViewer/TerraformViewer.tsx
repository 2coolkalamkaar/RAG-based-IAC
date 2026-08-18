"use client";

import { useState } from "react";
import { Download } from "lucide-react";
import styles from "./TerraformViewer.module.css";

interface Props {
  files: Record<string, string>;
}

export function TerraformViewer({ files }: Props) {
  const fileNames = Object.keys(files);
  const [active, setActive] = useState(fileNames[0] ?? "");

  if (fileNames.length === 0) return null;

  const activeCode = files[active] ?? "";

  function download(name: string, code: string) {
    const blob = new Blob([code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.tabBar}>
        {fileNames.map((name) => (
          <button
            key={name}
            className={`${styles.tab} ${name === active ? styles.tabActive : ""}`}
            onClick={() => setActive(name)}
          >
            📄 {name}
          </button>
        ))}
        <button
          className={styles.downloadBtn}
          onClick={() => download(active, activeCode)}
          title={`Download ${active}`}
        >
          <Download size={14} />
        </button>
      </div>
      <div className={styles.codeWrapper}>
        <pre className={styles.code}>
          <code>{activeCode}</code>
        </pre>
      </div>
    </div>
  );
}
