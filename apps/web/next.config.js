/** @type {import('next').NextConfig} */

// Where the FastAPI backend listens. Matches .env's API_PORT by default;
// override with API_PROXY_TARGET when that port is taken.
//
// Next bakes rewrites into the build output, so for `next start` the variable
// must be set when running `next build`. `next dev` reads it at startup.
const API_PROXY_TARGET = process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000';

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${API_PROXY_TARGET}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
