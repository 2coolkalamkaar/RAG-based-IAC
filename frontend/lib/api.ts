// API client — all calls to the FastAPI backend

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Job {
  id: string;
  thread_id: string;
  created_at: string;
  workflow: string;
  prompt: string;
  trust_score: number | null;
  trust_label: string | null;
}

export interface JobDetail extends Job {
  files: Record<string, string>;
}

export interface InternalDoc {
  filename: string;
  injected_at: string;
  description: string;
}

export interface HealthResponse {
  status: string;
  chunk_count: number;
  timestamp: string;
}

// ── Health ────────────────────────────────────────────────────────────────────
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

// ── Jobs ──────────────────────────────────────────────────────────────────────
export async function getJobs(limit = 50): Promise<Job[]> {
  const res = await fetch(`${API_BASE}/api/jobs?limit=${limit}`, {
    cache: "no-store",
  });
  const data = await res.json();
  return data.jobs;
}

export async function getJob(id: string): Promise<JobDetail> {
  const res = await fetch(`${API_BASE}/api/jobs/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Job not found");
  return res.json();
}

export async function deleteJob(id: string): Promise<void> {
  await fetch(`${API_BASE}/api/jobs/${id}`, { method: "DELETE" });
}

// ── HitL Actions ──────────────────────────────────────────────────────────────
export async function submitHitLAction(
  threadId: string,
  workflow: string,
  action: "approve" | "patch",
  patchRequest = "",
  prompt = ""
): Promise<void> {
  await fetch(`${API_BASE}/api/hitl/action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      workflow,
      action,
      patch_request: patchRequest,
      prompt,
    }),
  });
}

// ── Docs ──────────────────────────────────────────────────────────────────────
export async function uploadDoc(
  file: File,
  description: string
): Promise<{ chunks_added: number; filename: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("description", description);
  const res = await fetch(`${API_BASE}/api/docs/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function listDocs(limit = 20): Promise<InternalDoc[]> {
  const res = await fetch(`${API_BASE}/api/docs/list?limit=${limit}`, {
    cache: "no-store",
  });
  const data = await res.json();
  return data.docs;
}
