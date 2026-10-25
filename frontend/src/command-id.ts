type BrowserCrypto = {
  randomUUID?: () => string
  getRandomValues?: <T extends ArrayBufferView>(array: T) => T
}

let sequence = 0

/** Generate an idempotency key on HTTP LAN pages as well as HTTPS/localhost. */
export function createCommandId(cryptoApi: BrowserCrypto | undefined = globalThis.crypto): string {
  try {
    if (typeof cryptoApi?.randomUUID === "function") return cryptoApi.randomUUID()
    if (typeof cryptoApi?.getRandomValues === "function") {
      const bytes = cryptoApi.getRandomValues(new Uint8Array(16))
      return `cmd-${Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("")}`
    }
  } catch {
    // Some browsers expose crypto on an insecure origin but reject individual
    // methods. The command id isn't an authentication secret, so a collision-
    // resistant local fallback is sufficient for request idempotency.
  }
  sequence = (sequence + 1) >>> 0
  return `cmd-${Date.now().toString(36)}-${sequence.toString(36)}-${Math.random().toString(36).slice(2)}`
}
