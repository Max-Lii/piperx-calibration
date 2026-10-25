'use client';

import { useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { CodeBlock, Pre, type CodeBlockProps } from 'fumadocs-ui/components/codeblock';

/** Keep Fumadocs highlighting and add copying on plain HTTP preview origins. */
export function TutorialCodeBlock(props: CodeBlockProps) {
  const container = useRef<HTMLElement>(null);
  const [status, setStatus] = useState<'idle' | 'copied' | 'manual'>('idle');
  const [manualText, setManualText] = useState('');

  async function copy() {
    const code = container.current?.querySelector('pre code')?.cloneNode(true) as HTMLElement | undefined;
    if (!code) return;
    code.querySelectorAll('.nd-copy-ignore, [data-copy-ignore]').forEach(node => node.remove());
    const text = code.textContent ?? '';
    const active = document.activeElement as HTMLElement | null;
    let copied = false;
    if (navigator.clipboard?.writeText) {
      try { await navigator.clipboard.writeText(text); copied = true; } catch { /* Try HTTP-compatible selection below. */ }
    }
    if (!copied) {
      const field = document.createElement('textarea');
      field.value = text;
      field.setAttribute('readonly', '');
      field.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none';
      document.body.appendChild(field);
      field.select();
      try { copied = document.execCommand('copy'); } catch { copied = false; }
      field.remove();
      active?.focus({ preventScroll: true });
    }
    setStatus(copied ? 'copied' : 'manual');
    if (!copied) setManualText(text);
  }

  return <CodeBlock {...props} ref={container} allowCopy={false} Actions={({ className }) => (
    <div className={className}><button type="button" className="code-copy" onClick={copy} aria-label="Copy code / 复制代码">
      {status === 'copied' ? <Check size={16} /> : <Copy size={16} />}
      <span>{status === 'copied' ? 'Copied / 已复制' : 'Copy'}</span>
    </button></div>
  )}>
    <Pre>{props.children}</Pre>
    {status === 'manual' && <div className="manual-copy" role="status">
      <label>Automatic copy unavailable. Select and press Ctrl/Cmd+C. / 请选中后手动复制。
        <textarea readOnly value={manualText} onFocus={event => event.currentTarget.select()} />
      </label>
    </div>}
  </CodeBlock>;
}
