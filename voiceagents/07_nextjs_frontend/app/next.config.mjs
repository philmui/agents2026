import path from "node:path";
import { fileURLToPath } from "node:url";

// __dirname is not defined in ES modules, so we rebuild it from import.meta.url.
// This gives us the absolute path to THIS app folder to anchor the "@" alias.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Next.js configuration.
 *
 * `reactStrictMode` double-invokes some code in development to help you catch
 * bugs; it does NOT run in production builds.
 *
 * The `webpack` hook below registers the "@/..." import alias so that
 * `@/lib/token` means "<this app folder>/lib/token". We also declare the same
 * alias in tsconfig.json (`paths`) so the TypeScript editor understands it;
 * setting it here as well guarantees the bundler resolves it during
 * `next build`, independent of how tsconfig is parsed.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "@": __dirname,
    };
    return config;
  },
};

export default nextConfig;
