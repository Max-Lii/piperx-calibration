import { Heading } from 'fumadocs-ui/components/heading';
import { Callout } from 'fumadocs-ui/components/callout';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { TutorialCodeBlock } from './code-block';
import { ZoomableImage } from './zoomable-image';
import { Mermaid } from './mermaid';
import { resolveLink, withBasePath } from '@/lib/paths';
import imageAssets from '@/data/image_assets.json';
import { StaticCode, StaticMath, StaticStyles } from './static-content';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps, ReactNode } from 'react';
function Figure({ src, alt, caption }: { src: string; alt: string; caption?: string }) {
  const url = withBasePath(src);
  const size = imageAssets[src as keyof typeof imageAssets];
  return <figure><ZoomableImage src={url} alt={alt} caption={caption} width={size?.width} height={size?.height} />{caption && <figcaption>{caption}</figcaption>}</figure>;
}
function Anchor(props: ComponentProps<'a'>) {
  return <a {...props} href={typeof props.href === 'string' ? resolveLink(props.href) : props.href} />;
}
function Image(props: ComponentProps<'img'>) {
  return <img {...props} src={typeof props.src === 'string' ? withBasePath(props.src) : props.src} />;
}
function Table(props: ComponentProps<'table'>) {
  return <div className="relative overflow-auto prose-no-margin my-6" tabIndex={0}><table {...props} /></div>;
}
function FigureGrid({ children }: { children: ReactNode }) {
  return <div className="figure-grid">{children}</div>;
}
function ProcessFlow({ items }: { items: { title: string; text: string }[] }) {
  return <ol className="process-flow">{items.map((item, index) => (
    <li key={item.title}><span className="process-number" aria-hidden="true">{index + 1}</span><div><strong>{item.title}</strong><p>{item.text}</p></div></li>
  ))}</ol>;
}
export function getMDXComponents(components?: MDXComponents) {
  return {
    h1: (props) => <Heading as="h1" {...props} />,
    h2: (props) => <Heading as="h2" {...props} />,
    h3: (props) => <Heading as="h3" {...props} />,
    h4: (props) => <Heading as="h4" {...props} />,
    h5: (props) => <Heading as="h5" {...props} />,
    h6: (props) => <Heading as="h6" {...props} />,
    table: Table, Callout, pre: TutorialCodeBlock, Tab, Tabs, Figure, FigureGrid,
    ProcessFlow, Mermaid, StaticCode, StaticMath, StaticStyles, a: Anchor, img: Image,
    ...components,
  } satisfies MDXComponents;
}
export const useMDXComponents = getMDXComponents;
