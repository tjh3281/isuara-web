/**
 * Wires camera -> MediaPipe -> normalization -> classifier, and exposes the
 * result as React state.
 *
 * This is the web counterpart of the CameraX/ImageAnalysis block in
 * CameraScreen.kt. The differences are all forced by the platform: a
 * requestAnimationFrame loop replaces the analyzer executor, and the loop
 * deliberately runs the whole extract-and-predict chain inline rather than
 * queueing frames. There is no backpressure valve in the browser equivalent of
 * STRATEGY_KEEP_ONLY_LATEST, so processing synchronously is what keeps the
 * pipeline from building an unbounded backlog under load — a slow machine drops
 * frames instead, which is the behaviour we want.
 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react'

import { EmotionClassifier, type EmotionReading } from '../lib/emotion/classifier'
import { LandmarkExtractor } from '../lib/landmarkExtractor'
import { loadLabelMap, type LabelMap } from '../lib/labelMap'
import { PredictClient, type ConnectionState } from '../lib/predictClient'
import { SignPredictor } from '../lib/signPredictor'
import { fetchHealth, type Health } from '../lib/translateClient'

export type FacingMode = 'user' | 'environment'

export interface PipelineStatus {
  phase: 'idle' | 'starting' | 'running' | 'error'
  message: string
}

export function useSignPipeline() {
  // Typed without the explicit `| null` so the ref stays assignable to a JSX
  // `ref` prop under React 18's typings.
  const videoRef = useRef<HTMLVideoElement>(null)

  const clientRef = useRef<PredictClient | null>(null)
  const predictorRef = useRef<SignPredictor | null>(null)
  const extractorRef = useRef<LandmarkExtractor | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef<number | null>(null)

  const [connection, setConnection] = useState<ConnectionState>('closed')
  const [status, setStatus] = useState<PipelineStatus>({ phase: 'idle', message: '' })
  const [labelMap, setLabelMap] = useState<LabelMap | null>(null)
  const [facingMode, setFacingMode] = useState<FacingMode>('user')
  const [fps, setFps] = useState(0)
  /** null until probed, and stays null when no backend is reachable. */
  const [health, setHealth] = useState<Health | null>(null)
  /** The latest facial-expression reading, or null before one settles. */
  const [emotion, setEmotion] = useState<EmotionReading | null>(null)

  const emotionRef = useRef<EmotionClassifier | null>(null)
  if (!emotionRef.current) emotionRef.current = new EmotionClassifier()

  // Built once and kept for the life of the tab. The predictor holds the frame
  // window and sentence buffer, so recreating it would silently reset both.
  if (!clientRef.current) clientRef.current = new PredictClient(setConnection)
  if (!predictorRef.current) predictorRef.current = new SignPredictor(clientRef.current)

  const predictor = predictorRef.current
  const state = useSyncExternalStore(predictor.subscribe, predictor.getState)

  useEffect(() => {
    loadLabelMap().then(setLabelMap).catch((e) => {
      console.error('[pipeline] label map failed to load', e)
    })
  }, [])

  const stopStream = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }, [])

  const start = useCallback(
    async (mode: FacingMode) => {
      setStatus({ phase: 'starting', message: 'Requesting camera…' })
      stopStream()

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          // Matches the resolution the Android build asks CameraX for; the
          // browser treats these as hints and may hand back something close.
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: mode },
          audio: false,
        })
        streamRef.current = stream

        const video = videoRef.current
        if (!video) throw new Error('video element not mounted')
        video.srcObject = stream
        await video.play()

        if (!extractorRef.current) {
          setStatus({ phase: 'starting', message: 'Loading vision models…' })
          extractorRef.current = await LandmarkExtractor.create()
        }

        // Loaded alongside, not awaited: the expression model is ~15MB and the
        // camera should not wait on it. Sampling simply returns null until it
        // is ready.
        void emotionRef.current?.load().catch((e) => {
          console.warn('[emotion] classifier unavailable', e)
        })

        // Open the prediction socket only when a backend is actually there.
        //
        // Everything above this line — camera, MediaPipe, normalization — is
        // browser-local and works either way. When the site is served
        // statically (GitHub Pages) there is no /api, so probing first avoids
        // an endless reconnect loop against an origin that will never answer,
        // and the UI simply runs without gloss recognition. With the backend
        // running locally, this is unchanged: it connects and predicts.
        const probe = await fetchHealth().catch(() => null)
        setHealth(probe)
        if (probe) clientRef.current!.connect()

        setStatus({ phase: 'running', message: '' })

        let framesThisSecond = 0
        let lastFpsAt = performance.now()

        const loop = () => {
          rafRef.current = requestAnimationFrame(loop)

          const extractor = extractorRef.current
          if (!extractor || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return

          try {
            const { features } = extractor.extract(video, performance.now())
            predictor.onLandmarksExtracted(features)

            // Expression sampling is throttled inside the classifier and skips
            // while a previous inference is running, so this never blocks the
            // frame loop even though the model is far heavier than the sign one.
            void emotionRef.current
              ?.sample(video, video.videoWidth, video.videoHeight, features)
              .then((reading) => {
                if (reading) setEmotion(reading)
              })
          } catch (e) {
            console.warn('[pipeline] frame dropped', e)
          }

          framesThisSecond++
          const now = performance.now()
          if (now - lastFpsAt >= 1000) {
            setFps(framesThisSecond)
            framesThisSecond = 0
            lastFpsAt = now
          }
        }
        rafRef.current = requestAnimationFrame(loop)
      } catch (e) {
        const message =
          e instanceof DOMException && e.name === 'NotAllowedError'
            ? 'Camera permission denied. Allow camera access and reload.'
            : e instanceof Error
              ? e.message
              : String(e)
        setStatus({ phase: 'error', message })
      }
    },
    [predictor, stopStream],
  )

  const stop = useCallback(() => {
    stopStream()
    clientRef.current?.close()
    setStatus({ phase: 'idle', message: '' })
    setFps(0)
  }, [stopStream])

  const switchCamera = useCallback(() => {
    const next: FacingMode = facingMode === 'user' ? 'environment' : 'user'
    setFacingMode(next)
    if (status.phase === 'running') void start(next)
  }, [facingMode, start, status.phase])

  useEffect(() => {
    return () => {
      stopStream()
      clientRef.current?.close()
      extractorRef.current?.close()
    }
  }, [stopStream])

  return {
    videoRef,
    predictor,
    state,
    labelMap,
    connection,
    status,
    fps,
    facingMode,
    health,
    emotion,
    start: () => start(facingMode),
    stop,
    switchCamera,
  }
}
