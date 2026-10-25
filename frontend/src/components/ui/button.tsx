import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "../../lib/utils"

const variants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white disabled:pointer-events-none disabled:opacity-45",
  { variants: {
    variant: {
      default: "bg-zinc-900 text-zinc-50 hover:bg-zinc-800",
      secondary: "bg-zinc-100 text-zinc-900 hover:bg-zinc-200",
      danger: "bg-red-600 text-white hover:bg-red-700",
      outline: "border border-zinc-200 bg-white text-zinc-900 hover:bg-zinc-100",
    },
    size: { default: "h-10 px-4 py-2", sm: "h-8 px-3", lg: "h-11 px-6" },
  }, defaultVariants: { variant: "default", size: "default" } },
)

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof variants> { asChild?: boolean }

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, asChild, ...props }, ref) => {
  const Component = asChild ? Slot : "button"
  return <Component ref={ref} className={cn(variants({ variant, size }), className)} {...props} />
})
Button.displayName = "Button"
