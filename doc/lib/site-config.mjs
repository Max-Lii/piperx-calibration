import { readFileSync } from 'node:fs';

const defaults = JSON.parse(readFileSync(new URL('../site.config.json', import.meta.url), 'utf8'));

/** Local root by default; GitHub Actions supplies the project name itself. */
export function resolveSiteConfig(environment = process.env) {
  const repository = environment.GITHUB_ACTIONS === 'true' ? environment.GITHUB_REPOSITORY : undefined;
  const name = repository?.split('/')[1];
  const pagesBase = name && !name.endsWith('.github.io') ? `/${name}` : '';
  const basePath = environment.NEXT_PUBLIC_BASE_PATH ?? pagesBase;
  if (basePath && (!basePath.startsWith('/') || basePath.endsWith('/'))) {
    throw new Error('NEXT_PUBLIC_BASE_PATH must start with / and have no trailing slash');
  }
  return {
    basePath,
    repositoryUrl: (environment.NEXT_PUBLIC_REPOSITORY_URL
      || (repository ? `https://github.com/${repository}` : defaults.repositoryUrl)).replace(/\/$/, ''),
    repositoryRef: environment.NEXT_PUBLIC_REPOSITORY_REF || defaults.repositoryRef,
  };
}
