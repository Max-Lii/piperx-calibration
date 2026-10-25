import defaults from '@/site.config.json';

/** Root-relative MDX figures work both locally and on GitHub project Pages. */
export function withBasePath(path: string): string {
  return path.startsWith('/') && !path.startsWith('//')
    ? `${process.env.NEXT_PUBLIC_BASE_PATH || ''}${path}` : path;
}

/** MDX uses repository-relative links, independent of owner or Pages mount. */
export function resolveLink(path: string): string {
  if (!path.startsWith('repo:')) return withBasePath(path);
  const repository = (process.env.NEXT_PUBLIC_REPOSITORY_URL || defaults.repositoryUrl).replace(/\/$/, '');
  const ref = process.env.NEXT_PUBLIC_REPOSITORY_REF || defaults.repositoryRef;
  const relative = path.slice('repo:'.length).replace(/^\/+/, '');
  return `${repository}/blob/${encodeURIComponent(ref)}/${relative.split('/').map(encodeURIComponent).join('/')}`;
}
