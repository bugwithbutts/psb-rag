import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ask,
  checkHealth,
  closeConversation,
  getConversationId,
  loadConversation,
  resetConversation,
  rotateConversationId,
} from './api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  latencyMs?: number
  streaming?: boolean
  pending?: boolean
  status?: string
}

const WELCOME: Message = {
  id: 'welcome',
  role: 'assistant',
  content: 'Здравствуйте! Я — юридический ассистент ПСБ. Задайте вопрос.',
  timestamp: Date.now(),
}

let counter = 0
function uid(): string {
  counter += 1
  return `m-${Date.now()}-${counter}`
}

type HealthState = 'starting' | 'ready' | 'error'

/** Пока backend строит локальный индекс, порт уже открыт — просто ждём. */
function useHealth(): { state: HealthState; detail?: string } {
  const [state, setState] = useState<HealthState>('starting')
  const [detail, setDetail] = useState<string | undefined>()

  useEffect(() => {
    let alive = true
    let timer: number | undefined

    const poll = async () => {
      const health = await checkHealth()
      if (!alive) return
      if (health.agent) {
        setState('ready')
        setDetail(undefined)
        return
      }
      if (health.status === 'error') {
        setState('error')
        setDetail(health.detail)
        return
      }
      setState('starting')
      timer = window.setTimeout(poll, 2000)
    }

    void poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  return { state, detail }
}

function useChat(ready: boolean) {
  const [messages, setMessages] = useState<Message[]>([WELCOME])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const conversationId = useRef('')
  if (!conversationId.current) conversationId.current = getConversationId()

  useEffect(() => () => abortRef.current?.abort(), [])

  // После F5 сервер помнит диалог — восстанавливаем сообщения.
  useEffect(() => {
    let alive = true
    loadConversation(conversationId.current)
      .then((turns) => {
        if (!alive || turns.length === 0) return
        const restored = turns.map((turn, index) => ({
          id: `h-${index}`,
          role: turn.role,
          content: turn.content,
          timestamp: Date.now(),
        }))
        // Не затираем диалог, если пользователь успел что-то отправить.
        setMessages((prev) => (prev.length > 1 ? prev : [WELCOME, ...restored]))
      })
      .catch(() => undefined)
    return () => {
      alive = false
    }
  }, [])

  // Закрытие вкладки: просим сервер удалить историю.
  useEffect(() => {
    const onPageHide = (event: PageTransitionEvent) => {
      if (!event.persisted) closeConversation()
    }
    window.addEventListener('pagehide', onPageHide)
    return () => window.removeEventListener('pagehide', onPageHide)
  }, [])

  const send = useCallback(
    async (raw: string) => {
      const query = raw.trim()
      if (!query || isStreaming || !ready) return

      const assistantId = uid()
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: 'user', content: query, timestamp: Date.now() },
        {
          id: assistantId,
          role: 'assistant',
          content: '',
          timestamp: Date.now(),
          pending: true,
          streaming: true,
        },
      ])

      const controller = new AbortController()
      abortRef.current = controller
      setIsStreaming(true)

      const patch = (updater: (m: Message) => Message) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? updater(m) : m)))

      let finished = false
      try {
        for await (const ev of ask(query, conversationId.current, controller.signal)) {
          if (ev.type === 'status') {
            patch((m) => ({ ...m, status: ev.message }))
          } else if (ev.type === 'token') {
            patch((m) => ({
              ...m,
              content: m.content + ev.text,
              status: undefined,
              pending: false,
            }))
          } else if (ev.type === 'done') {
            finished = true
            patch((m) => ({
              ...m,
              content: ev.answer,
              latencyMs: ev.latencyMs,
              streaming: false,
              pending: false,
              status: undefined,
            }))
          } else if (ev.type === 'error') {
            throw new Error(ev.message)
          }
        }
        if (!finished) patch((m) => ({ ...m, streaming: false, pending: false }))
      } catch (err) {
        const aborted = err instanceof Error && err.name === 'AbortError'
        const text = aborted
          ? 'Генерация остановлена.'
          : `Ошибка: ${err instanceof Error ? err.message : 'нет ответа'}`
        patch((m) => ({
          ...m,
          content: m.content ? `${m.content}\n\n${text}` : text,
          streaming: false,
          pending: false,
          status: undefined,
        }))
      } finally {
        setIsStreaming(false)
        abortRef.current = null
      }
    },
    [isStreaming, ready],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    void resetConversation(conversationId.current)
    conversationId.current = rotateConversationId()
    setMessages([WELCOME])
  }, [])

  return { messages, isStreaming, send, stop, clear }
}

function Header({
  onClear,
  isStreaming,
  health,
}: {
  onClear: () => void
  isStreaming: boolean
  health: HealthState
}) {
  const label =
    health === 'error'
      ? 'ошибка запуска'
      : health === 'starting'
        ? 'запуск…'
        : isStreaming
          ? 'обработка…'
          : 'готов'
  const pillClass = [
    'status-pill',
    health === 'ready' && isStreaming ? 'busy' : '',
    health === 'starting' ? 'booting' : '',
    health === 'error' ? 'failed' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <header className="header">
      <div className="header-brand">
        <div className="brand-logo" aria-hidden="true">
          <span>ПСБ</span>
        </div>
        <div className="brand-text">
          <h1>Юридический ассистент</h1>
        </div>
      </div>
      <div className="header-controls">
        <span className={pillClass}>
          <span className="status-dot" />
          {label}
        </span>
        <button type="button" className="btn-ghost" onClick={onClear} title="Очистить чат">
          Очистить
        </button>
      </div>
    </header>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* буфер обмена недоступен */
    }
  }

  return (
    <article className={`msg ${isUser ? 'msg-user' : 'msg-assistant'}`}>
      {!isUser && (
        <div className="avatar" aria-hidden="true">
          ПСБ
        </div>
      )}
      <div className="bubble">
        <div className="bubble-body">{message.content || 'Думаю…'}</div>
        <div className="bubble-meta">
          {!isUser && message.latencyMs !== undefined && !message.streaming && (
            <span className="meta-latency">{(message.latencyMs / 1000).toFixed(1)} с</span>
          )}
          <span className="meta-time">
            {new Date(message.timestamp).toLocaleTimeString('ru-RU', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {!isUser && !message.pending && !message.streaming && (
            <button type="button" className="meta-action" onClick={copy}>
              {copied ? 'скопировано' : 'копировать'}
            </button>
          )}
        </div>
      </div>
    </article>
  )
}

function TypingIndicator({ label }: { label?: string }) {
  return (
    <div className="typing">
      <div className="typing-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <span>{label ?? 'Думаю…'}</span>
    </div>
  )
}

function Chat({ messages }: { messages: Message[] }) {
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const last = messages[messages.length - 1]
  const lastPending = last?.pending === true
  const visible = messages.filter((m) => !(m.pending && !m.content))

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, lastPending])

  return (
    <main className="chat" aria-live="polite">
      <div className="chat-inner">
        {visible.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {lastPending ? <TypingIndicator label={last?.status} /> : null}
        <div ref={bottomRef} />
      </div>
    </main>
  )
}

function Composer({
  onSend,
  onStop,
  isStreaming,
  ready,
}: {
  onSend: (text: string) => void
  onStop: () => void
  isStreaming: boolean
  ready: boolean
}) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  const submit = () => {
    const text = value.trim()
    if (!text || isStreaming || !ready) return
    onSend(text)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const autoResize = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }

  return (
    <div className="composer-wrap">
      <div className="composer">
        <div className="composer-row">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            placeholder={ready ? 'Задайте вопрос…' : 'Модель инициализируется…'}
            onChange={(e) => {
              setValue(e.target.value)
              autoResize()
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            disabled={isStreaming || !ready}
          />
          {isStreaming ? (
            <button
              type="button"
              className="btn-send btn-stop"
              onClick={onStop}
              title="Остановить"
            >
              ⏹
            </button>
          ) : (
            <button
              type="button"
              className="btn-send"
              onClick={submit}
              disabled={!value.trim() || !ready}
              title="Отправить"
            >
              ➤
            </button>
          )}
        </div>
        <div className="composer-hint">
          {ready
            ? 'Enter — отправить · Shift+Enter — новая строка'
            : 'Первый запуск занимает около 1–2 минут — дождитесь готовности'}
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const { state: health, detail } = useHealth()
  const { messages, isStreaming, send, stop, clear } = useChat(health === 'ready')

  return (
    <div className="app">
      <Header onClear={clear} isStreaming={isStreaming} health={health} />
      {health !== 'ready' && (
        <div className={`boot-banner${health === 'error' ? ' boot-error' : ''}`} role="status">
          {health === 'starting' && <span className="boot-spinner" aria-hidden="true" />}
          <div className="boot-text">
            <strong>
              {health === 'error'
                ? 'Модель не инициализирована'
                : 'Инициализация модели…'}
            </strong>
            <span>
              {health === 'error'
                ? detail ?? 'Перезапустите контейнеры и обновите страницу.'
                : 'Первый запуск занимает около 1–2 минут. Вопросы станут доступны автоматически.'}
            </span>
          </div>
        </div>
      )}
      <Chat messages={messages} />
      <Composer
        onSend={send}
        onStop={stop}
        isStreaming={isStreaming}
        ready={health === 'ready'}
      />
      <footer className="app-footer">
        Ответы формируются по внутренней базе знаний. Проверяйте ссылки на источники.
      </footer>
    </div>
  )
}
