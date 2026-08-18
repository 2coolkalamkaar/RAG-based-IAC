import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Terraform Architect | AI-Powered Infrastructure Generator",
  description:
    "Self-healing agentic RAG pipeline for Terraform generation — grounding, validation, and human-in-the-loop trust gates.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
