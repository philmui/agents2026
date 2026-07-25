// Ambient type declarations so a plain `tsc --noEmit` (and your editor) accept
// side-effect CSS imports like `import "./globals.css"`. Next.js understands
// these natively at build/dev time; this file just teaches the bare TypeScript
// compiler the same thing, so type-checking is clean on a fresh clone.
declare module "*.css";
