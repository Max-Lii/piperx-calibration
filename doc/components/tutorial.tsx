import Link from 'next/link';
import { source } from '@/lib/source';
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from 'fumadocs-ui/layouts/docs/page';
import { getMDXComponents } from '@/components/mdx';
import { ThemeSwitch } from 'fumadocs-ui/layouts/shared/slots/theme-switch';
import { RepositoryLink } from '@/components/repository-link';

export function Tutorial({ language = 'en' }: { language?: 'en' | 'zh' }) {
  const page = source.getPage(language === 'zh' ? ['zh'] : []);
  if (!page) throw new Error(`Missing ${language} tutorial`);
  const Content = page.data.body;
  return <DocsPage lang={language} tableOfContent={{ style: 'block' }} tableOfContentPopover={{ style: 'block' }} toc={page.data.toc} breadcrumb={{ enabled: false }} footer={{ enabled: false }}>
    <div className="tutorial-toolbar"><nav aria-label="Language" className="language-switch">
      <Link href="/" prefetch={false} lang="en" hrefLang="en" aria-current={language === 'en' ? 'page' : undefined}>English</Link>
      <Link href="/zh/" prefetch={false} lang="zh-CN" hrefLang="zh-CN" aria-current={language === 'zh' ? 'page' : undefined}>中文</Link>
    </nav><RepositoryLink language={language} /><ThemeSwitch mode="light-dark-system" /></div>
    <DocsTitle>{page.data.title}</DocsTitle>
    <DocsDescription>{page.data.description}</DocsDescription>
    <DocsBody><Content components={getMDXComponents()} /></DocsBody>
  </DocsPage>;
}
