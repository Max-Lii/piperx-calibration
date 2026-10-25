import { defineConfig } from 'fumadocs-mdx/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkMdxMermaid } from 'fumadocs-core/mdx-plugins';
import { rehypeStaticContent } from './lib/rehype-static-content.mjs';
export default defineConfig({
  mdxOptions: { remarkPlugins: [remarkMath, remarkMdxMermaid], rehypePlugins: (plugins) => [rehypeKatex, ...plugins, rehypeStaticContent] },
});
