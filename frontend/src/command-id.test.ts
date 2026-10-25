import { describe, expect, it } from "vitest"
import { createCommandId } from "./command-id"

describe("createCommandId", () => {
  it("uses randomUUID when the browser provides it", () => {
    expect(createCommandId({ randomUUID: () => "known-uuid" })).toBe("known-uuid")
  })

  it("works on an insecure HTTP origin without randomUUID", () => {
    const first = createCommandId({})
    const second = createCommandId({})
    expect(first).toMatch(/^cmd-/)
    expect(first.length).toBeGreaterThanOrEqual(8)
    expect(second).not.toBe(first)
  })

  it("uses getRandomValues when available", () => {
    const id = createCommandId({
      getRandomValues: (array) => {
        new Uint8Array(array.buffer, array.byteOffset, array.byteLength).fill(7)
        return array
      },
    })
    expect(id).toBe(`cmd-${"07".repeat(16)}`)
  })
})
