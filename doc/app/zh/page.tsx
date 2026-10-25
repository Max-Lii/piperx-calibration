import type { Metadata } from 'next';
import { Tutorial } from '@/components/tutorial';
export const metadata: Metadata = { title: 'Piper-X 校准教程', description: '从机器人模型到工作台校准，串联原理、代码与验证。' };
export default function Page() {
  return <Tutorial language="zh" />;
}
