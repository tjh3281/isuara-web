/**
 * The "Text → Sign" tab — ports avatar/ui/AvatarPlayerScreen.kt.
 *
 * A full-bleed 3D viewport with a single input bar over it. Type a word or a
 * sentence and the avatar signs it: an exact catalog hit plays that clip, and
 * anything multi-word is synthesised by joining clips with co-articulation
 * bridges. Signs play once — looping would read as a stutter rather than a sign.
 *
 * Drag to orbit, wheel or pinch to zoom, matching TouchOrbitController.kt.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { AvatarRenderer } from '../lib/avatar/renderer'
import { MotionPlayer } from '../lib/avatar/motionPlayer'
import { CATALOG, loadMotion, resolveQuery } from '../lib/avatar/repository'

/** Loaded paused on open, as the Android screen does. */
const INITIAL_SIGN = 'terima_kasih'

export function AvatarScreen() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rendererRef = useRef<AvatarRenderer | null>(null)
  const playerRef = useRef<MotionPlayer | null>(null)
  const rafRef = useRef<number | null>(null)

  const [input, setInput] = useState('')
  const [activeWord, setActiveWord] = useState('')
  const [status, setStatus] = useState('')

  if (!playerRef.current) playerRef.current = new MotionPlayer()
  const player = playerRef.current

  // ── renderer lifecycle ──
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const renderer = new AvatarRenderer(canvas)
    rendererRef.current = renderer

    let last = performance.now()
    const loop = () => {
      rafRef.current = requestAnimationFrame(loop)
      const now = performance.now()
      // Clamped exactly as the Kotlin does: a backgrounded tab would otherwise
      // return a multi-second delta and skip the whole clip on resume.
      const delta = Math.min(Math.max((now - last) / 1000, 0.001), 0.1)
      last = now
      renderer.render(player.advanceTime(delta))
    }
    rafRef.current = requestAnimationFrame(loop)

    const observer = new ResizeObserver(() => renderer.resize())
    observer.observe(canvas)

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      observer.disconnect()
      renderer.dispose()
      rendererRef.current = null
    }
  }, [player])

  // ── initial clip, paused ──
  useEffect(() => {
    let cancelled = false
    loadMotion(INITIAL_SIGN).then((motion) => {
      if (cancelled || !motion) return
      player.isLooping = false
      player.setMotion(motion, false)
      setActiveWord(motion.word)
    })
    return () => {
      cancelled = true
    }
  }, [player])

  const play = useCallback(
    async (query: string) => {
      const effective = query.trim()
      if (!effective) return
      setStatus('')

      const resolved = await resolveQuery(effective)
      if (!resolved) {
        setStatus(`No sign for “${effective}”`)
        return
      }
      player.isLooping = false
      player.setMotion(resolved.motion, true)
      setActiveWord(resolved.motion.word)
    },
    [player],
  )

  // ── orbit / zoom ──
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let dragging = false
    let lastX = 0
    let lastY = 0

    const down = (e: PointerEvent) => {
      dragging = true
      lastX = e.clientX
      lastY = e.clientY
      canvas.setPointerCapture(e.pointerId)
    }
    const move = (e: PointerEvent) => {
      if (!dragging) return
      const dx = e.clientX - lastX
      const dy = e.clientY - lastY
      lastX = e.clientX
      lastY = e.clientY
      // Dragging right turns the avatar right; dragging down raises the camera,
      // which is the direction fixed in the Android "correct vertical orbit" commit.
      rendererRef.current?.orbit(-dx * 0.01, dy * 0.005)
    }
    const up = (e: PointerEvent) => {
      dragging = false
      canvas.releasePointerCapture(e.pointerId)
    }
    const wheel = (e: WheelEvent) => {
      e.preventDefault()
      rendererRef.current?.zoom(e.deltaY > 0 ? 1.08 : 0.92)
    }

    canvas.addEventListener('pointerdown', down)
    canvas.addEventListener('pointermove', move)
    canvas.addEventListener('pointerup', up)
    canvas.addEventListener('pointercancel', up)
    canvas.addEventListener('wheel', wheel, { passive: false })
    return () => {
      canvas.removeEventListener('pointerdown', down)
      canvas.removeEventListener('pointermove', move)
      canvas.removeEventListener('pointerup', up)
      canvas.removeEventListener('pointercancel', up)
      canvas.removeEventListener('wheel', wheel)
    }
  }, [])

  return (
    <div className="avatar">
      <canvas ref={canvasRef} className="avatar__canvas" />

      {/*
        Just the sign being played. The fixed camera presets that used to sit
        here are gone — dragging and pinching reach every angle they offered,
        and a row of five buttons over the viewport was the loudest thing on a
        screen whose subject is the avatar.
      */}
      <div className="avatar__hud">
        <span className="avatar__word">{activeWord || '—'}</span>
      </div>

      {status && <p className="avatar__status">{status}</p>}

      <form
        className="avatar__bar"
        onSubmit={(e) => {
          e.preventDefault()
          void play(input)
        }}
      >
        <input
          className="avatar__input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter word or sentence"
          list="avatar-vocabulary"
          enterKeyHint="go"
          autoComplete="off"
        />
        {/* Native suggestions rather than a custom dropdown: the catalog is 33
            entries and this keeps the bar as minimal as the phone's. */}
        <datalist id="avatar-vocabulary">
          {CATALOG.map((item) => (
            <option key={item.key} value={item.title}>
              {item.translation}
            </option>
          ))}
        </datalist>
        <button type="submit" className="avatar__send" title="Play sign">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor" aria-hidden="true">
            <path d="M4 11h12.17l-5.59-5.59L12 4l8 8-8 8-1.41-1.41L16.17 13H4z" />
          </svg>
          <span className="sr-only">Play sign</span>
        </button>
      </form>
    </div>
  )
}
