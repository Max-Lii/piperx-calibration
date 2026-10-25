import type { Config } from "tailwindcss"

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#101927",
        panel: "#182536",
        accent: "#1269d3",
      },
      boxShadow: { panel: "0 8px 30px rgba(15, 23, 42, 0.08)" },
    },
  },
  plugins: [],
} satisfies Config
