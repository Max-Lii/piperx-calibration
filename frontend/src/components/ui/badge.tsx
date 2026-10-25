import { cn } from "../../lib/utils"

export function Badge({ ok, children, className }: { ok?: boolean; children: React.ReactNode; className?: string }) {
  return <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium", ok === true && "border-emerald-200 bg-emerald-50 text-emerald-700", ok === false && "border-red-200 bg-red-50 text-red-700", ok === undefined && "border-zinc-200 bg-zinc-100 text-zinc-600", className)}>{children}</span>
}
