import type { Metadata } from 'next';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { Provider } from '@/components/provider';
import { source } from '@/lib/source';
import './global.css';
export const metadata: Metadata = { title: 'Piper-X Calibration', description: 'A practical guide to robot, camera, and workcell calibration.' };
export default function Layout({ children }: { children: React.ReactNode }) {
  return <html lang="en" suppressHydrationWarning><body><Provider>
    <DocsLayout tree={source.getPageTree()} sidebar={{ enabled: false }} nav={{ title: 'Piper-X Calibration', url: '/' }}>
      {children}
    </DocsLayout>
  </Provider></body></html>;
}
