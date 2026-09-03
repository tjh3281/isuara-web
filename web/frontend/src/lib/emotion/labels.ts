/**
 * The eight AffectNet expression classes — ports emotion/EmotionLabel.kt.
 *
 * Array order IS the model's output index order, taken verbatim from
 * `idx_to_emotion_class` in EmotiEffLib's `facial_analysis.py`. Reordering these
 * entries silently mislabels every prediction, so the order is part of the model
 * contract rather than a stylistic choice.
 *
 * Each entry carries what the downstream stages need: the descriptor goes into
 * the translator prompt, the style directive to the voice engine, and pitch and
 * rate drive the Web Speech fallback.
 */

export interface EmotionLabel {
  /** Circumplex arousal, 0 (calm) to 1 (activated). */
  arousal: number
  /** Circumplex valence, -1 (negative) to +1 (positive). */
  valence: number
  descriptorMs: string
  descriptorEn: string
  /** Absolute pitch for the Web Speech fallback; 1.0 is normal. */
  pitch: number
  /**
   * Multiplier on the base speech rate rather than an absolute rate, so the
   * app's deliberate "slightly slow for clarity" baseline survives.
   */
  rateScale: number
  /**
   * Deliberately English even though the spoken text is Malay — Google's
   * guidance is that style prompts work best in English regardless of target.
   */
  styleDirective: string
}

/** Order is the model contract. Do not sort. */
export const EMOTION_LABELS: readonly EmotionLabel[] = [
  {
    arousal: 0.85, valence: -0.7,
    descriptorMs: 'marah', descriptorEn: 'anger',
    pitch: 0.92, rateScale: 1.12,
    styleDirective: 'Say this angrily and firmly, with clipped, forceful delivery.',
  },
  {
    arousal: 0.45, valence: -0.4,
    descriptorMs: 'meluat', descriptorEn: 'contempt',
    pitch: 0.95, rateScale: 0.95,
    styleDirective: 'Say this with cold, dismissive disdain.',
  },
  {
    // Below the high-arousal threshold on purpose: disgust is intense but not
    // urgent, and should not trigger the shouted register fear and anger do.
    arousal: 0.55, valence: -0.6,
    descriptorMs: 'jijik', descriptorEn: 'disgust',
    pitch: 0.9, rateScale: 0.95,
    styleDirective: 'Say this with revulsion, as if recoiling from something.',
  },
  {
    arousal: 0.9, valence: -0.8,
    descriptorMs: 'takut', descriptorEn: 'fear',
    pitch: 1.18, rateScale: 1.25,
    styleDirective: 'Say this urgently and fearfully, fast, with a strained, breathless voice.',
  },
  {
    arousal: 0.7, valence: 0.8,
    descriptorMs: 'gembira', descriptorEn: 'happiness',
    pitch: 1.12, rateScale: 1.05,
    styleDirective: 'Say this warmly and brightly, with an audible smile.',
  },
  {
    arousal: 0.25, valence: 0,
    descriptorMs: 'neutral', descriptorEn: 'neutral',
    pitch: 1, rateScale: 1,
    styleDirective: 'Say this in a calm, clear, matter-of-fact voice.',
  },
  {
    arousal: 0.3, valence: -0.6,
    descriptorMs: 'sedih', descriptorEn: 'sadness',
    pitch: 0.88, rateScale: 0.82,
    styleDirective: 'Say this softly and heavily, slowly, with a downcast tone.',
  },
  {
    arousal: 0.8, valence: 0.2,
    descriptorMs: 'terkejut', descriptorEn: 'surprise',
    pitch: 1.2, rateScale: 1.1,
    styleDirective: 'Say this with sudden surprise, sharp and raised in pitch.',
  },
]

export const EMOTION_COUNT = 8
export const HIGH_AROUSAL_THRESHOLD = 0.6

/** Index of NEUTRAL in the contract order. */
export const NEUTRAL_INDEX = 5

/**
 * The single predicate gating both the colloquial register and the prosody
 * boost. Defined once so the two cannot drift apart.
 */
export function isHighArousal(label: EmotionLabel): boolean {
  return label.arousal >= HIGH_AROUSAL_THRESHOLD
}

/** The label for a model output index, or null when out of range. */
export function labelFromIndex(index: number): EmotionLabel | null {
  return EMOTION_LABELS[index] ?? null
}
