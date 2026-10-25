import { createHash } from 'node:crypto';
import { toHtml } from 'hast-util-to-html';

/** Keep build-rendered code and math opaque to React's navigation payload.
 * The DOM, MathML and highlighting remain intact; individual static tokens no
 * longer need a separate React element in every exported RSC representation.
 */
export function rehypeStaticContent() {
  return (tree) => {
    const palette = new Map();
    function compactStyles(node) {
      if (node.type === 'element' && node.properties?.style?.includes('--shiki-light:')) {
        const style = node.properties.style;
        const name = `syntax-${createHash('sha256').update(style).digest('hex').slice(0, 10)}`;
        palette.set(name, style);
        const classes = node.properties.className ?? [];
        node.properties.className = [...(Array.isArray(classes) ? classes : classes.split(' ')), name];
        delete node.properties.style;
      }
      for (const child of node.children ?? []) compactStyles(child);
    }
    function staticNode(name, html, className) {
      return {
        type: 'mdxJsxTextElement', name,
        attributes: [
          { type: 'mdxJsxAttribute', name: 'html', value: html },
          ...(className ? [{ type: 'mdxJsxAttribute', name: 'className', value: className }] : []),
        ],
        children: [],
      };
    }
    function visit(parent) {
      if (!parent.children) return;
      parent.children = parent.children.map((node) => {
        const classes = node.properties?.className ?? [];
        const isMath = node.type === 'element' && node.tagName === 'span' && classes.includes('katex');
        const isCode = node.type === 'element' && node.tagName === 'code'
          && node.children.some((child) => child.type === 'element');
        if (isMath || isCode) {
          const extra = Object.keys(node.properties ?? {}).filter((key) => !['className', 'class'].includes(key));
          if (extra.length) throw new Error(`Static ${node.tagName}: unsupported properties ${extra.join(', ')}`);
          if (isCode) compactStyles(node);
          const className = node.properties?.className ?? node.properties?.class;
          return staticNode(isMath ? 'StaticMath' : 'StaticCode', toHtml(node.children),
            Array.isArray(className) ? className.join(' ') : className);
        }
        visit(node);
        return node;
      });
    }
    visit(tree);
    if (palette.size) {
      const style = staticNode('StaticStyles', [...palette].map(([name, value]) => `.${name}{${value}}`).join('\n'));
      style.type = 'mdxJsxFlowElement';
      tree.children.unshift(style);
    }
  };
}
