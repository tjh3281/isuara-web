/**
 * Vocabulary catalog, clip loading and sentence synthesis — ports
 * avatar/data/MotionRepository.kt.
 *
 * The catalog is transcribed from the Kotlin rather than derived from the
 * filenames, because it carries the display titles, English glosses and
 * categories the UI matches against — none of which the files themselves know.
 *
 * Clips are fetched over HTTP and cached: each is 100-300KB, and a sentence
 * replays the same words repeatedly.
 */

import { lerp, parseMotion, type BimFrame, type BimMotion, type HandJoints } from './motion'

export interface VocabularyItem {
  key: string
  title: string
  translation: string
  category: string
  isSentence?: boolean
  assetFileName: string
}

export const CATALOG: VocabularyItem[] = [
  // Sentences
  { key: 'apa_khabar_suhu', title: 'Apa Khabar → Suhu', translation: 'Sentence: How are you → Temperature', category: 'Continuous Sentences', isSentence: true, assetFileName: 'apa_khabar_suhu_3d.json' },
  { key: 'suhu_apa_khabar', title: 'Suhu → Apa Khabar', translation: 'Sentence: Temperature → How are you', category: 'Continuous Sentences', isSentence: true, assetFileName: 'suhu_apa_khabar_3d.json' },

  // 10 audited pilot vocabulary
  { key: 'terima_kasih', title: 'Terima Kasih', translation: 'Thank you', category: 'Pilot Benchmarks', assetFileName: 'terima_kasih.json' },
  { key: 'makan', title: 'Makan', translation: 'Eat', category: 'Pilot Benchmarks', assetFileName: 'makan.json' },
  { key: 'gembira', title: 'Gembira', translation: 'Happy', category: 'Pilot Benchmarks', assetFileName: 'gembira.json' },
  { key: 'cinta', title: 'Cinta', translation: 'Love', category: 'Pilot Benchmarks', assetFileName: 'cinta.json' },
  { key: 'belajar', title: 'Belajar', translation: 'Study / Learn', category: 'Pilot Benchmarks', assetFileName: 'belajar.json' },
  { key: 'doktor', title: 'Doktor', translation: 'Doctor', category: 'Pilot Benchmarks', assetFileName: 'doktor.json' },
  { key: 'baju', title: 'Baju', translation: 'Clothes', category: 'Pilot Benchmarks', assetFileName: 'baju.json' },
  { key: 'air', title: 'Air', translation: 'Water', category: 'Pilot Benchmarks', assetFileName: 'air.json' },
  { key: 'abang', title: 'Abang', translation: 'Brother', category: 'Pilot Benchmarks', assetFileName: 'abang.json' },
  { key: 'apa_khabar', title: 'Apa Khabar', translation: 'How are you', category: 'Pilot Benchmarks', assetFileName: 'apa_khabar_3d.json' },
  { key: 'suhu', title: 'Suhu', translation: 'Temperature', category: 'Pilot Benchmarks', assetFileName: 'suhu_3d.json' },

  // Common conversation and greetings
  { key: 'hai', title: 'Hai', translation: 'Hi / Hello', category: 'Daily Conversation', assetFileName: 'hai.json' },
  { key: 'hello', title: 'Hello', translation: 'Hello', category: 'Daily Conversation', assetFileName: 'hello.json' },
  { key: 'saya', title: 'Saya', translation: 'I / Me', category: 'Daily Conversation', assetFileName: 'saya.json' },
  { key: 'awak', title: 'Awak', translation: 'You', category: 'Daily Conversation', assetFileName: 'awak.json' },
  { key: 'nama', title: 'Nama', translation: 'Name', category: 'Daily Conversation', assetFileName: 'nama.json' },
  { key: 'tolong', title: 'Tolong', translation: 'Help / Please', category: 'Daily Conversation', assetFileName: 'tolong.json' },
  { key: 'sama_sama', title: 'Sama-sama', translation: "You're welcome", category: 'Daily Conversation', assetFileName: 'sama_sama.json' },
  { key: 'baik', title: 'Baik', translation: 'Good / Fine', category: 'Daily Conversation', assetFileName: 'baik.json' },
  { key: 'cantik', title: 'Cantik', translation: 'Beautiful', category: 'Daily Conversation', assetFileName: 'cantik.json' },
  { key: 'ibu', title: 'Ibu', translation: 'Mother', category: 'Family', assetFileName: 'ibu.json' },
  { key: 'bapa', title: 'Bapa', translation: 'Father', category: 'Family', assetFileName: 'bapa.json' },
  { key: 'sekolah', title: 'Sekolah', translation: 'School', category: 'Places', assetFileName: 'sekolah.json' },
  { key: 'hospital', title: 'Hospital', translation: 'Hospital', category: 'Places', assetFileName: 'hospital.json' },

  // Numbers and letters
  { key: '1', title: '1 (Satu)', translation: 'One', category: 'Numbers', assetFileName: '1.json' },
  { key: '2', title: '2 (Dua)', translation: 'Two', category: 'Numbers', assetFileName: '2.json' },
  { key: '3', title: '3 (Tiga)', translation: 'Three', category: 'Numbers', assetFileName: '3.json' },
  { key: '4', title: '4 (Empat)', translation: 'Four', category: 'Numbers', assetFileName: '4.json' },
  { key: '5', title: '5 (Lima)', translation: 'Five', category: 'Numbers', assetFileName: '5.json' },
  { key: 'a', title: 'Huruf A', translation: 'Letter A', category: 'Alphabet', assetFileName: 'a.json' },
  { key: 'b', title: 'Huruf B', translation: 'Letter B', category: 'Alphabet', assetFileName: 'b.json' },
  { key: 'c', title: 'Huruf C', translation: 'Letter C', category: 'Alphabet', assetFileName: 'c.json' },
]

/** Frames inserted between two words so the hands travel rather than jump. */
const BRIDGE_FRAMES = 16
const SYNTHESIS_FPS = 50

const cache = new Map<string, BimMotion>()

function assetUrl(fileName: string): string {
  return `${import.meta.env.BASE_URL}motions/${fileName}`
}

/** Loads a clip by vocabulary key, or null when there is no such sign. */
export async function loadMotion(key: string): Promise<BimMotion | null> {
  const cached = cache.get(key)
  if (cached) return cached

  const item = CATALOG.find((v) => v.key.toLowerCase() === key.toLowerCase())
  const fileName = item?.assetFileName ?? (key.endsWith('.json') ? key : `${key}.json`)

  try {
    const response = await fetch(assetUrl(fileName))
    if (!response.ok) return null
    const motion = parseMotion(await response.json())
    cache.set(key, motion)
    return motion
  } catch (e) {
    console.warn(`[avatar] could not load motion "${key}" (${fileName})`, e)
    return null
  }
}

/** Vocabulary matching a query in title, English gloss or key. */
export function searchVocabulary(query: string): VocabularyItem[] {
  if (!query.trim()) return CATALOG
  const q = query.trim().toLowerCase()
  return CATALOG.filter(
    (v) =>
      v.title.toLowerCase().includes(q) ||
      v.translation.toLowerCase().includes(q) ||
      v.key.toLowerCase().includes(q),
  )
}

/**
 * Joins several words into one continuous clip.
 *
 * Signs are not concatenated end to end: a 16-frame co-articulation bridge is
 * interpolated between each pair, because real signing moves continuously
 * between shapes and a hard cut reads as two separate signs rather than a
 * sentence.
 */
export async function synthesizeSentence(words: string[]): Promise<BimMotion | null> {
  const loaded = await Promise.all(words.map((w) => loadMotion(w)))
  const motions = loaded.filter((m): m is BimMotion => m !== null)

  if (motions.length === 0) return null
  if (motions.length === 1) return motions[0]

  const frames: BimFrame[] = []
  let index = 0

  motions.forEach((motion, i) => {
    for (const f of motion.frames) {
      frames.push({ ...f, frame: index, time: index / SYNTHESIS_FPS })
      index++
    }

    const next = motions[i + 1]
    if (!next) return

    const last = motion.frames[motion.frames.length - 1]
    const first = next.frames[0]
    if (!last || !first) return

    for (let b = 1; b <= BRIDGE_FRAMES; b++) {
      const alpha = b / (BRIDGE_FRAMES + 1)
      frames.push({
        frame: index,
        time: index / SYNTHESIS_FPS,
        pose: {
          ...last.pose,
          nose:
            last.pose.nose && first.pose.nose
              ? lerp(last.pose.nose, first.pose.nose, alpha)
              : last.pose.nose,
        },
        leftHand: bridgeHand(last.leftHand, first.leftHand, alpha),
        rightHand: bridgeHand(last.rightHand, first.rightHand, alpha),
      })
      index++
    }
  })

  const title = words
    .map((w) => w.replace(/_/g, ' '))
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')

  return {
    word: title,
    fps: SYNTHESIS_FPS,
    numFrames: frames.length,
    duration: index / SYNTHESIS_FPS,
    frames,
  }
}

function bridgeHand(
  h0: HandJoints | null,
  h1: HandJoints | null,
  alpha: number,
): HandJoints | null {
  if (!h0 || !h1) return h0 ?? h1
  if (h0.points.length !== 21 || h1.points.length !== 21) return h0
  return {
    active: h0.active || h1.active,
    points: h0.points.map((p, i) => lerp(p, h1.points[i], alpha)),
  }
}

/**
 * Resolves free text to a clip: exact catalog hit, else multi-word synthesis,
 * else the best partial match. Mirrors triggerPlay in AvatarPlayerScreen.kt.
 */
export async function resolveQuery(
  query: string,
): Promise<{ key: string; motion: BimMotion } | null> {
  const raw = query.trim()
  if (!raw) return null
  const clean = raw.toLowerCase().replace(/\s+/g, '_')

  const direct = CATALOG.find(
    (v) => v.key.toLowerCase() === clean || v.title.toLowerCase() === raw.toLowerCase(),
  )
  if (direct) {
    const motion = await loadMotion(direct.key)
    if (motion) return { key: direct.key, motion }
  }

  const parts = raw.toLowerCase().split(/\s+/)
  if (parts.length > 1) {
    const synthesized = await synthesizeSentence(parts)
    if (synthesized) return { key: clean, motion: synthesized }
  }

  const fallback = searchVocabulary(raw)[0]
  if (fallback) {
    const motion = await loadMotion(fallback.key)
    if (motion) return { key: fallback.key, motion }
  }

  return null
}
