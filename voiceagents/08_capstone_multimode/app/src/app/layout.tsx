// The root layout wraps every page. In the Next.js App Router, this file is
// required. We load the Inter font (the course's typeface) and our global CSS.
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// next/font downloads Inter at build time and serves it locally (no runtime
// request to Google), which is fast and privacy-friendly.
const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "Voice Agents Capstone — Transcribe / Translate / Assist",
  description:
    "Module 08 capstone: one web app unifying three OpenAI Realtime modes plus a tool-calling voice assistant.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
