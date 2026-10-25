/** HTML generated from trusted MDX by build-time Shiki/KaTeX, never user input. */
export function StaticCode({ html, className }: { html: string; className?: string }) {
  return <code className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

export function StaticMath({ html, className }: { html: string; className?: string }) {
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

export function StaticStyles({ html }: { html: string }) {
  return <style dangerouslySetInnerHTML={{ __html: html }} />;
}
