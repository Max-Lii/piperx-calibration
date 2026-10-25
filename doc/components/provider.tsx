'use client';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { usePathname } from 'next/navigation';
import { useEffect } from 'react';
import type { ReactNode } from 'react';
export function Provider({ children }: { children: ReactNode }) {
  const chinese = /\/zh\/?$/.test(usePathname() ?? '/');
  useEffect(() => { document.documentElement.lang = chinese ? 'zh-CN' : 'en'; }, [chinese]);
  return <RootProvider search={{ enabled: false }} i18n={{ locale: chinese ? 'zh' : 'en', translations: chinese ? { 'On this page(table of contents)': '本页目录' } : {} }}>{children}</RootProvider>;
}
