// =============================================================================
// components/StatusPill.tsx  -  a tiny colored badge showing connection state.
// =============================================================================
// Shared by all three panels so the app looks consistent. Pure presentation:
// it takes a status string and renders a colored pill. No logic, no state.
// =============================================================================

"use client";

// The pill is green when live, red on error, and neutral otherwise. We decide
// the color from the text so callers can pass any human-readable status.
export function StatusPill({ status }: { status: string }) {
  const s = status.toLowerCase();

  // Pick a CSS modifier class (defined in globals.css) from the status text.
  let variant = "";
  if (s.includes("error") || s.includes("fail")) variant = "error";
  else if (
    s.includes("connected") ||
    s.includes("live") ||
    s.includes("listening") ||
    s.includes("hearing")
  ) {
    variant = "live";
  }

  return <span className={`status ${variant}`}>{status}</span>;
}
