/**
 * Bottom tab bar — ports the NavigationBar in MainActivity.kt.
 *
 * Two destinations, same labels and icons as the app: "Sign → Voice" on a
 * Videocam icon, "Text → Sign" on AccessibilityNew. The selected pill uses
 * Material 3's indicator, which is why the active item has a filled capsule
 * behind the icon rather than just a colour change.
 */

export type Tab = 'camera' | 'avatar'

interface Props {
  tab: Tab
  onChange: (tab: Tab) => void
}

export function BottomNav({ tab, onChange }: Props) {
  return (
    <nav className="nav" role="tablist">
      <button
        type="button"
        role="tab"
        aria-selected={tab === 'camera'}
        className={`nav__item${tab === 'camera' ? ' nav__item--active' : ''}`}
        onClick={() => onChange('camera')}
      >
        <span className="nav__indicator">
          {/* Icons.Filled.Videocam */}
          <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">
            <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z" />
          </svg>
        </span>
        <span className="nav__label">Sign → Voice</span>
      </button>

      <button
        type="button"
        role="tab"
        aria-selected={tab === 'avatar'}
        className={`nav__item${tab === 'avatar' ? ' nav__item--active' : ''}`}
        onClick={() => onChange('avatar')}
      >
        <span className="nav__indicator">
          {/* Icons.Filled.AccessibilityNew */}
          <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">
            <circle cx="12" cy="4" r="2" />
            <path d="M19 13v-2c-1.54.02-3.09-.75-4.07-1.83l-1.29-1.43c-.17-.19-.38-.34-.61-.45-.01 0-.01-.01-.02-.01H13c-.35-.2-.75-.3-1.19-.26C10.76 7.11 10 8.04 10 9.09V15c0 1.1.9 2 2 2h5v5h2v-5.5c0-1.1-.9-2-2-2h-3v-3.45c1.29 1.07 3.25 1.94 5 1.95zm-9 7c-1.66 0-3-1.34-3-3 0-1.31.84-2.41 2-2.83V12.1a5 5 0 1 0 5.9 5.9h-2.07c-.41 1.16-1.52 2-2.83 2z" />
          </svg>
        </span>
        <span className="nav__label">Text → Sign</span>
      </button>
    </nav>
  )
}
