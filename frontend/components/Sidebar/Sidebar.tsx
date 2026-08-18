"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, History, BookOpen, Cpu, Shield, Zap, GitBranch, HardHat } from "lucide-react";
import styles from "./Sidebar.module.css";

const NAV_LINKS = [
  { href: "/", icon: LayoutDashboard, label: "Dashboard" },
  { href: "/history", icon: History, label: "Job History" },
  { href: "/knowledge", icon: BookOpen, label: "Knowledge Base" },
];

const WORKFLOWS = [
  { id: "basic", icon: Cpu, label: "Basic LLM", color: "#8b949e" },
  { id: "rag", icon: Zap, label: "Standard RAG", color: "#58a6ff" },
  { id: "advanced", icon: GitBranch, label: "Advanced RAG", color: "#bc8cff" },
  { id: "secure", icon: Shield, label: "Secure RAG", color: "#3fb950" },
  { id: "hitl", icon: HardHat, label: "HitL RAG", color: "#ffa657" },
];

interface SidebarProps {
  selectedWorkflow: string;
  onWorkflowChange: (id: string) => void;
  chunkCount?: number;
}

export function Sidebar({ selectedWorkflow, onWorkflowChange, chunkCount }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>🏗️</div>
        <div>
          <div className={styles.logoTitle}>Terraform Architect</div>
          <div className={styles.logoSub}>AI Infrastructure Platform</div>
        </div>
      </div>

      <nav className={styles.nav}>
        {NAV_LINKS.map(({ href, icon: Icon, label }) => (
          <Link
            key={href}
            href={href}
            className={`${styles.navLink} ${pathname === href ? styles.active : ""}`}
          >
            <Icon size={16} />
            {label}
          </Link>
        ))}
      </nav>

      <div className={styles.section}>
        <div className={styles.sectionLabel}>Workflow</div>
        {WORKFLOWS.map(({ id, icon: Icon, label, color }) => (
          <button
            key={id}
            className={`${styles.workflowBtn} ${selectedWorkflow === id ? styles.workflowActive : ""}`}
            onClick={() => onWorkflowChange(id)}
          >
            <Icon size={14} style={{ color }} />
            {label}
          </button>
        ))}
      </div>

      {chunkCount !== undefined && (
        <div className={styles.footer}>
          <div className={styles.footerStat}>
            <span className={styles.footerLabel}>Knowledge Base</span>
            <span className={styles.footerValue}>{chunkCount.toLocaleString()} chunks</span>
          </div>
          <div className={styles.footerStat}>
            <span className={styles.footerLabel}>AWS Provider Docs</span>
            <span className={styles.footerValue}>1,696 docs</span>
          </div>
        </div>
      )}
    </aside>
  );
}
