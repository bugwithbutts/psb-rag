// Клиент бэкенда: SSE-поток чата, история диалога и id вкладки.

export type AskEvent =
  | { type: 'meta'; conversationId: string }
  | { type: 'status'; message: string }
  | { type: 'token'; text: string }
  | { type: 'done'; answer: string; latencyMs: number }
  | { type: 'error'; message: string }

export interface Turn {
  role: 'user' | 'assistant'
  content: string
}

export interface HealthStatus {
  status: 'starting' | 'ok' | 'error'
  agent: boolean
  detail?: string
}

// --- id диалога: свой у каждой вкладки, переживает F5 -----------------------

const KEY = 'psb.conversationId'
const TAB_MARKER = 'psb-tab'

function newId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `c-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function getConversationId(): string {
  // window.name переживает F5, но не копируется при дублировании вкладки.
  if (window.name !== TAB_MARKER) {
    sessionStorage.removeItem(KEY)
    window.name = TAB_MARKER
  }
  let id = sessionStorage.getItem(KEY)
  if (!id) {
    id = newId()
    sessionStorage.setItem(KEY, id)
  }
  return id
}

export function rotateConversationId(): string {
  const id = newId()
  sessionStorage.setItem(KEY, id)
  return id
}

/** Сообщить серверу о закрытии вкладки (sendBeacon умеет только POST). */
export function closeConversation(): void {
  const id = sessionStorage.getItem(KEY)
  if (id) navigator.sendBeacon(`/api/conversations/${id}/close?ts=${Date.now()}`)
}

// --- SSE -------------------------------------------------------------------

function parseEvent(raw: string): AskEvent | null {
  const data = raw
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n')

  if (!data || data === '[DONE]') return null
  try {
    return JSON.parse(data) as AskEvent
  } catch {
    return null
  }
}

const RETRY_STATUSES = new Set([502, 503, 504])
const RETRY_ATTEMPTS = 10
const RETRY_DELAY_MS = 1000

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function postChat(
  query: string,
  conversationId: string,
  signal?: AbortSignal,
): Promise<Response> {
  const body = JSON.stringify({ query, conversation_id: conversationId, ts: Date.now() })
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body,
      signal,
    })
    // Nginx отдаёт 502/503/504, пока backend ещё поднимается после рестарта.
    if (!RETRY_STATUSES.has(res.status) || attempt >= RETRY_ATTEMPTS) return res
    await sleep(RETRY_DELAY_MS)
  }
}

/** Проверить готовность backend (индекс и агент). */
export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch('/api/health', { cache: 'no-store' })
    if (!res.ok) return { status: 'starting', agent: false }
    return (await res.json()) as HealthStatus
  } catch {
    return { status: 'starting', agent: false }
  }
}

export async function* ask(
  query: string,
  conversationId: string,
  signal?: AbortSignal,
): AsyncGenerator<AskEvent> {
  const res = await postChat(query, conversationId, signal)

  if (!res.ok || !res.body) throw new Error(`API ответил статусом ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ''

      for (const block of blocks) {
        const event = parseEvent(block)
        if (event) yield event
        else if (block.includes('[DONE]')) return
      }
    }

    buffer += decoder.decode()
    if (buffer.trim()) {
      const event = parseEvent(buffer)
      if (event) yield event
      else if (buffer.includes('[DONE]')) return
    }
  } finally {
    reader.releaseLock()
  }
}

export async function loadConversation(id: string): Promise<Turn[]> {
  const res = await fetch(`/api/conversations/${id}?ts=${Date.now()}`)
  if (!res.ok) return []
  const data = (await res.json()) as { turns?: Turn[] }
  return data.turns ?? []
}

export async function resetConversation(id: string): Promise<void> {
  await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
}
