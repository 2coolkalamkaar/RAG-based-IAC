"use client";

import { useEffect, useRef } from "react";
import styles from "./LogTerminal.module.css";

export interface LogEntry {
  ts: string;
  msg: string;
  level: "info" | "ok" | "warn" | "err" | "dim";
}

interface Props {
  entries: LogEntry[];
}

export function LogTerminal({ entries }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  if (entries.length === 0) return null;

  return (
    <div className={styles.terminal}>
      {entries.map((e, i) => (
        <div key={i} className={styles.line}>
          <span className={styles.ts}>[{e.ts}]</span>{" "}
          <span className={`${styles.msg} ${styles[e.level]}`}>{e.msg}</span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
