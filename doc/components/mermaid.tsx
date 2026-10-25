import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import { renderMermaidSVG } from 'beautiful-mermaid';

/** Server-rendered Mermaid SVG using the renderer recommended by Fumadocs. */
export function Mermaid({ chart }: { chart: string }) {
  try {
    const svg = renderMermaidSVG(chart, {
      bg: 'var(--color-fd-background)',
      fg: 'var(--color-fd-foreground)',
      line: 'var(--color-fd-muted-foreground)',
      accent: 'var(--color-fd-primary)',
      muted: 'var(--color-fd-muted-foreground)',
      surface: 'var(--color-fd-card)',
      border: 'var(--color-fd-border)',
      transparent: true,
      interactive: true,
      font: 'ui-sans-serif, system-ui, sans-serif',
    });
    return <div className="mermaid-diagram" dangerouslySetInnerHTML={{ __html: svg }} />;
  } catch {
    return <CodeBlock title="Mermaid"><Pre>{chart}</Pre></CodeBlock>;
  }
}
