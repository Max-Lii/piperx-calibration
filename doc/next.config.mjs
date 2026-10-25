import { createMDX } from 'fumadocs-mdx/next';
import { resolveSiteConfig } from './lib/site-config.mjs';
const site = resolveSiteConfig();
export default createMDX()({
  output: 'export',
  trailingSlash: true,
  basePath: site.basePath,
  env: {
    NEXT_PUBLIC_BASE_PATH: site.basePath,
    NEXT_PUBLIC_REPOSITORY_URL: site.repositoryUrl,
    NEXT_PUBLIC_REPOSITORY_REF: site.repositoryRef,
  },
  images: { unoptimized: true },
  reactStrictMode: true,
});
