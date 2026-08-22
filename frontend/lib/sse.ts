"use client";

import { useRef, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export type SSEEvent =
  | { event: "node_update"; node: string; status: string; [key: string]: unknown }
  | { event: "trust_score"; node: string; status: string; score: number; label: string; factors: Record<string, number>; explanation: string }
  | { event: "hitl_pause"; node: string; status: string; thread_id: string; files?: Record<string, string>; citations?: string[] }
  | { event: "plan_preview"; node: string; plan_summary?: Record<string, number>; cost_estimate_monthly?: number; blast_radius_passed?: boolean; cost_ceiling_passed?: boolean }
  | { event: "apply_result"; node: string; status?: string; apply_outputs?: Record<string, unknown> }
  | { event: "destroy_result"; node: string; status?: string }
  | { event: "complete"; files: Record<string, string>; citations: string[]; trust_score: number; trust_label: string; trust_factors: Record<string, number>; trust_explanation: string; cost_estimate: string }
  | { event: "code_stream"; chunk: string }
  | { event: "error"; message: string };

interface UseSSEStreamOptions {
  onEvent: (event: SSEEvent) => void;
  onDone?: () => void;
}

export function useSSEStream({ onEvent, onDone }: UseSSEStreamOptions) {
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(
    async (workflow: string, prompt: string, threadId?: string) => {
      // Cancel any existing stream
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch(`${API_BASE}/api/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workflow, prompt, thread_id: threadId }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          onEvent({ event: "error", message: `HTTP ${res.status}` });
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const parsed = JSON.parse(line.slice(6)) as SSEEvent;
                onEvent(parsed);
                if (parsed.event === "complete" || parsed.event === "hitl_pause") {
                  onDone?.();
                  return;
                }
              } catch {
                // malformed JSON line — skip
              }
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== "AbortError") {
          onEvent({ event: "error", message: err.message });
        }
      } finally {
        onDone?.();
      }
    },
    [onEvent, onDone]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { start, stop };
}
