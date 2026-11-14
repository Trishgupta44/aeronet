import { useTheme } from '../theme/useTheme.js';

const OPTIONS = [
  { mode: 'system', label: 'Match system theme' },
  { mode: 'light', label: 'Light theme' },
  { mode: 'dark', label: 'Dark theme' },
];

// 16 px line icons, drawn with currentColor so they follow the button.
const ICONS = {
  system: (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
      <rect x="1.5" y="2.5" width="13" height="8.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M5.5 14h5M8 11v3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  ),
  light: (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
      <circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M3.4 12.6l1.3-1.3M11.3 4.7l1.3-1.3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  ),
  dark: (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
      <path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
    </svg>
  ),
};

/** Three-way colour theme switch: match the system, light, dark. `onChrome` styles it for the always-navy bars. */
export default function ThemeToggle({ onChrome = false }) {
  const { mode, setMode } = useTheme();
  return (
    <div className={`theme-toggle${onChrome ? ' theme-toggle--chrome' : ''}`} role="radiogroup" aria-label="Colour theme">
      {OPTIONS.map((o) => (
        <button key={o.mode} type="button" role="radio" aria-checked={mode === o.mode} aria-label={o.label} title={o.label} onClick={() => setMode(o.mode)}>
          {ICONS[o.mode]}
        </button>
      ))}
    </div>
  );
}
