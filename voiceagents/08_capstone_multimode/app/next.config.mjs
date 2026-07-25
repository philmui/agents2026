// Next.js configuration. We keep it almost empty on purpose: the defaults are
// what we want for this course. Beginners can ignore this file.
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The README permits both localhost and 127.0.0.1. Next's dev server treats
  // those as different origins, so explicitly allow both for HMR and client JS.
  allowedDevOrigins: ["localhost", "127.0.0.1"],
};

export default nextConfig;
