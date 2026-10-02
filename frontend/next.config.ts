import type { NextConfig } from "next";

const API_URL = process.env.API_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const config: NextConfig = {
  reactStrictMode: true,
  output: "standalone",
  // Proxy the API through the Next server so the browser makes same-origin
  // requests: no CORS preflight on every PNG tile, and one URL to configure
  // when this is deployed behind a single hostname.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default config;
