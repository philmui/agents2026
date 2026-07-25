// =============================================================================
// app/layout.tsx  —  the ROOT layout for the Next.js App Router.
// =============================================================================
//
// In the App Router, every page is wrapped by this one file. It renders the
// <html> and <body> tags for the whole site and is the right place to:
//   - set the page <title> and description (the `metadata` export below),
//   - load a font once for the entire app,
//   - import the global stylesheet.
//
// There is exactly ONE root layout. `children` is whatever page is being shown
// (here, only app/page.tsx). This file runs on the server by default, which is
// fine: it renders static shell HTML and contains no browser-only code.
// =============================================================================

import type { Metadata } from "next";
import { Inter } from "next/font/google"; // Next.js self-hosts Google Fonts for us.
import "./globals.css"; // the global styles (pastel palette, layout). Imported once.

// Load the Inter font (the course's house font). `subsets` trims the file to the
// Latin characters we use; `display: "swap"` shows fallback text immediately and
// swaps Inter in when it loads, so the page never blocks on the font.
const inter = Inter({ subsets: ["latin"], display: "swap" });

// `metadata` is a special export the App Router reads to fill the <head>: the
// browser-tab title and the description search engines show.
export const metadata: Metadata = {
  title: "Voice Agents · Talk to gpt-realtime-2.1",
  description:
    "A minimal Next.js + React frontend that talks to OpenAI's gpt-realtime-2.1 in the browser over WebRTC.",
};

// The root layout component. Next.js passes the current page in as `children`.
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      {/* The Inter font class is applied to <body>, so every element inherits it. */}
      <body className={inter.className}>{children}</body>
    </html>
  );
}
