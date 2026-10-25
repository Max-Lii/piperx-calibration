import * as React from "react"
import { cn } from "../../lib/utils"

export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) =>
  <div ref={ref} className={cn("rounded-xl border border-zinc-200 bg-white shadow-sm", className)} {...props} />)
Card.displayName = "Card"

export const CardHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) =>
  <div className={cn("flex flex-col gap-1.5 p-4", className)} {...props} />
export const CardTitle = ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) =>
  <h3 className={cn("font-semibold text-zinc-950", className)} {...props} />
export const CardContent = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) =>
  <div className={cn("p-4 pt-0", className)} {...props} />
