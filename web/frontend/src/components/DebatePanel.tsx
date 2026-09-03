/**
 * The debate, shown in full underneath the translation.
 *
 * Every interpreter's answer is on screen from the moment its model replies,
 * with the judge's pick marked. Rows fill in individually rather than all at
 * once, so a slow model does not hold up the two that already answered.
 *
 * This used to be a collapsed accordion, on the reasoning that the finished
 * sentence is the point and the machinery behind it should not compete with it.
 * That was the wrong call for this app: the multi-model debate IS the thing
 * worth showing, and hiding it behind a control meant it was never seen. It is
 * open, always, with no toggle to find.
 */

import { shortModelName, STAGE_LABELS, type DebateProgress } from '../lib/translateClient'

interface Props {
  progress: DebateProgress
  /** True while the request is still running. */
  active: boolean
}

export function DebatePanel({ progress, active }: Props) {
  const { candidates, verdict, stage } = progress
  if (candidates.length === 0 && !active) return null

  const answered = candidates.filter((c) => c.sentence !== null).length
  const failed = candidates.filter((c) => c.failed).length

  const heading = active
    ? STAGE_LABELS[stage] || 'Consulting interpreters…'
    : `${answered} of ${candidates.length} interpreters answered`

  return (
    <section className="debate">
      <h3 className="debate__heading">
        {active && <span className="spinner" aria-hidden="true" />}
        {heading}
      </h3>

      <ol className="debate__agents">
        {candidates.map((c) => {
          const chosen = verdict?.choice === c.index
          return (
            <li
              key={c.index}
              className={`debate__agent${chosen ? ' debate__agent--chosen' : ''}`}
            >
              <span className="debate__model">
                {shortModelName(c.model)}
                {/* Marks the judge's pick without relying on the tint alone,
                    which a colour-blind reader would not see. */}
                {chosen && <span className="debate__chosen-tag">chosen</span>}
              </span>
              {c.failed ? (
                <span className="debate__failed">unavailable</span>
              ) : c.sentence ? (
                <span className="debate__sentence">{c.sentence}</span>
              ) : (
                <span className="debate__pending">thinking…</span>
              )}
            </li>
          )
        })}
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
    </section>
  )
}
