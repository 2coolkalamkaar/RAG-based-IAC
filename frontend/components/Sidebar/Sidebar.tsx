"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Zap, History, Settings, Database, Shield, LayoutDashboard } from "lucide-react";
import styles from "./Sidebar.module.css";

const NAV = [
  { href: "/",         icon: Zap,             label: "Generate" },
  { href: "/history",  icon: History,          label: "History" },
  { href: "/knowledge",icon: Database,         label: "Knowledge" },
  { href: "/settings", icon: Settings,         label: "Settings" },
];

interface Props {
  chunkCount?: number;
  awsConnected?: boolean;
  applyPaused?: boolean;
  onTogglePause?: () => void;
  selectedWorkflow?: string;
  onWorkflowChange?: (id: string) => void;
}

export function Sidebar({ chunkCount, awsConnected = false, applyPaused = false, onTogglePause, selectedWorkflow, onWorkflowChange }: Props) {
  const path = usePathname();

  return (
    <aside className={styles.sidebar}>
      {/* Logo */}
      <div className={styles.logo}>
        <div className={styles.logoIcon}>
          <Zap size={16} className={styles.logoZap} />
        </div>
        <div className={styles.logoText}>
          <span className={styles.logoName}>TerraForge</span>
          <span className={styles.logoTag}>IaC Agent</span>
        </div>
      </div>

      {/* Nav */}
      <nav className={styles.nav}>
        {NAV.map(({ href, icon: Icon, label }) => {
          const active = path === href;
          return (
            <Link key={href} href={href} className={`${styles.navItem} ${active ? styles.navActive : ""}`}>
              <Icon size={15} className={styles.navIcon} />
              <span className={styles.navLabel}>{label}</span>
              {active && <span className={styles.navActiveDot} />}
            </Link>
          );
        })}
      </nav>

      <div className={styles.spacer} />

      {/* Status section */}
      <div className={styles.statusSection}>
        {/* AWS status */}
        <div className={styles.statusRow}>
          <span className={`dot ${awsConnected ? "dot-green" : "dot-dim"} ${awsConnected ? "dot-pulse" : ""}`} />
          <span className={styles.statusLabel}>
            {awsConnected ? "AWS Connected" : "AWS Not Configured"}
          </span>
        </div>

        {/* KB chunks */}
        {chunkCount !== undefined && (
          <div className={styles.statusRow}>
            <span className="dot dot-blue" />
            <span className={styles.statusLabel}>
              {chunkCount.toLocaleString()} chunks indexed
            </span>
          </div>
        )}

        {/* Circuit breaker */}
        <div className={styles.circuitBreaker}>
          <Shield size={12} className={styles.shieldIcon} style={{ color: applyPaused ? "var(--red)" : "var(--text-muted)" }} />
          <span className={styles.statusLabel} style={{ color: applyPaused ? "var(--red)" : "var(--text-muted)" }}>
            {applyPaused ? "Apply Paused" : "Apply Enabled"}
          </span>
          {onTogglePause && (
            <button className={styles.pauseToggle} onClick={onTogglePause} title="Toggle circuit breaker">
              {applyPaused ? "Resume" : "Pause"}
            </button>
          )}
        </div>
      </div>

      {/* Version */}
      <div className={styles.version}>v2.0 · apply pipeline</div>
    </aside>
  );
}
