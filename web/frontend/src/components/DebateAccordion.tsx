/**
 * The debate, shown as a two-step accordion — ports the accordion added in
 * "feat(ui): show the debate as a two-step accordion".
 *
 * Step one lists the three interpreters, each row filling in the moment its
 * model answers rather than when the slowest one does. Step two is the judge's
 * pick and its stated reason.
 *
 * Collapsed by default: the point of the panel is the finished sentence, and the
 * machinery behind it is available to anyone curious without competing with the
 * result for attention.
 */

import { useState } from 'react'

import { shortModelName, STAGE_LABELS, type DebateProgress } from '../lib/translateClient'

interface Props {
  progress: DebateProgress
  /** True while the request is still running. */
  active: boolean
}

export function DebateAccordion({ progress, active }: Props) {
  const [open, setOpen] = useState(false)

  const { candidates, verdict, stage } = progress
  if (candidates.length === 0 && !active) return null

  const answered = candidates.filter((c) => c.sentence !== null).length
  const failed = candidates.filter((c) => c.failed).length

  const summary = active
    ? STAGE_LABELS[stage] || 'Consulting interpreters…'
    : `${answered} of ${candidates.length} interpreters answered`

  return (
    <div className="debate">
      <button
        type="button"
        className="debate__toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {active && <span className="spinner" aria-hidden="true" />}
        <span className="debate__summary">{summary}</span>
        <span className={`debate__chevron${open ? ' debate__chevron--open' : ''}`} aria-hidden="true">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
            <path d="M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z" />
          </svg>
        </span>
      </button>

      {open && (
        <div className="debate__body">
          <ol className="debate__agents">
            {candidates.map((c) => (
              <li
                key={c.index}
                className={`debate__agent${
                  verdict?.choice === c.index ? ' debate__agent--chosen' : ''
                }`}
              >
                <span className="debate__model">{shortModelName(c.model)}</span>
                {c.failed ? (
                  <span className="debate__failed">unavailable</span>
                ) : c.sentence ? (
                  <span className="debate__sentence">{c.sentence}</span>
                ) : (
                  <span className="debate__pending">thinking…</span>
                )}
              </li>
            ))}
          </ol>

          {verdict && (
            <p className="debate__verdict">
              <strong>Judge:</strong> {verdict.reason || 'chose the best candidate'}
            </p>
          )}

          {failed > 0 && (
            <p className="debate__note">
              {failed} model{failed > 1 ? 's' : ''} did not answer — the debate
              continues with the rest.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
