/**
 * The seam between the browser's feature pipeline and whatever runs the model.
 *
 * Everything upstream of this file — camera, MediaPipe, normalization — is
 * finished and browser-local. This is the only place that knows inference
 * happens elsewhere, which is deliberate: the model currently has to run
 * server-side because it was converted with SELECT_TF_OPS and needs the Flex
 * delegate, which no browser TFLite runtime provides. If the model is later
 * re-exported to TensorFlow.js, only this file changes.
 *
 * A WebSocket rather than POST-per-window: a running camera fires a prediction
 * roughly three times a second, and setup cost on each would dominate the ~2ms
 * of actual inference.
 */

import { wsUrl } from './apiBase'
import { FINAL_FEATURES, SEQUENCE_LENGTH } from './frameNormalizer'

export interface Prediction {
  label: string
  confidence: number
  isConfident: boolean
}

export type ConnectionState = 'connecting' | 'open' | 'closed'

const WINDOW_FLOATS = SEQUENCE_LENGTH * FINAL_FEATURES

function socketUrl(): string {
  return wsUrl('/ws/predict')
}

const MIN_RECONNECT_MS = 500
const MAX_RECONNECT_MS = 8000

export class PredictClient {
  private socket: WebSocket | null = null
  /** Resolvers for windows sent but not yet answered, in send order. */
  private pending: Array<(p: Prediction) => void> = []
  private reconnectDelay = MIN_RECONNECT_MS
  private closed = false
  /** Server-side faults are logged once per connection, not once per frame. */
  private loggedError = ''

  constructor(private onStateChange: (state: ConnectionState) => void = () => {}) {}

  connect(): void {
    this.closed = false
    this.open()
  }

  private open(): void {
    this.onStateChange('connecting')
    const socket = new WebSocket(socketUrl())
    socket.binaryType = 'arraybuffer'
    this.socket = socket

    socket.onopen = () => {
      // Deliberately does NOT reset the backoff. When the backend has no model
      // it accepts the socket, reports the fault and closes immediately — so an
      // open event is not evidence of a working connection. Resetting here made
      // the delay collapse back to 500ms on every attempt, producing a
      // reconnect twice a second for as long as the tab stayed open. The reset
      // lives in onmessage instead, where a real reply proves the link works.
      this.onStateChange('open')
    }

    socket.onmessage = (event) => {
      let payload: Prediction & { error?: string }
      try {
        payload = JSON.parse(event.data as string)
      } catch {
        console.warn('[predict] unparseable frame', event.data)
        return
      }

      // Replies arrive in send order, so the oldest waiter owns this one.
      const resolve = this.pending.shift()
      if (payload.error) {
        // Once per distinct fault, not once per frame: a missing model would
        // otherwise bury every other message in the console.
        if (this.loggedError !== payload.error) {
          this.loggedError = payload.error
          console.warn('[predict] server error:', payload.error)
        }
        return
      }
      // A real prediction is the only proof the link actually works, so this is
      // where the backoff resets.
      this.reconnectDelay = MIN_RECONNECT_MS
      this.loggedError = ''
      resolve?.(payload)
    }

    socket.onclose = () => {
      this.onStateChange('closed')
      // Anything still waiting will never be answered; drop the waiters rather
      // than leaving the predictor's in-flight flag stuck on.
      this.pending = []
      if (!this.closed) {
        setTimeout(() => this.open(), this.reconnectDelay)
        // Back off to 8s so a backend that is down — or up but without a
        // model — does not become a connection storm for as long as the tab
        // stays open.
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, MAX_RECONNECT_MS)
      }
    }

    socket.onerror = () => {
      // onclose always follows, and handles the retry.
      socket.close()
    }
  }

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN
  }

  /**
   * Sends one 30 x 780 window and resolves with the top class.
   *
   * Resolves to null when the socket is not open, so a caller can treat a
   * disconnected backend as "no prediction this window" rather than an error to
   * handle on every frame.
   */
  predict(features: Float32Array): Promise<Prediction | null> {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return Promise.resolve(null)
    }
    if (features.length !== WINDOW_FLOATS) {
      return Promise.reject(
        new Error(`expected ${WINDOW_FLOATS} floats, got ${features.length}`),
      )
    }

    return new Promise<Prediction | null>((resolve) => {
      this.pending.push(resolve)
      this.socket!.send(features.buffer as ArrayBuffer)
    })
  }

  close(): void {
    this.closed = true
    this.socket?.close()
    this.socket = null
    this.pending = []
  }
}
