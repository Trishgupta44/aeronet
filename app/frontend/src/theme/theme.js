// Colour theme: "system" (follow the operating system), "light" or "dark". The resolved theme is written to <html data-theme>,
// which the CSS tokens key on. The choice is kept in localStorage. ?theme=light|dark (or the older ?light) overrides it for one
// visit without saving it, so printed figures and screenshots stay reproducible.
// The same logic runs once in index.html before the first paint (so there is no flash); this module takes over after that.

const KEY = 'aeronet-theme';
const MODES = ['system', 'light', 'dark'];
const listeners = new Set();

const hasWindow = typeof window !== 'undefined';
const systemQuery = hasWindow && window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

function stored() {
  try {
    const v = window.localStorage.getItem(KEY);
    return MODES.includes(v) ? v : null;
  } catch {
    return null; // private window, blocked storage
  }
}

function forcedByAddress() {
  if (!hasWindow) return null;
  const q = new URLSearchParams(window.location.search);
  const t = q.get('theme');
  if (t === 'light' || t === 'dark') return t;
  return q.has('light') ? 'light' : null;
}

const forced = forcedByAddress();
let mode = forced ?? (hasWindow ? stored() : null) ?? 'system';
let snapshot = { mode, resolved: 'light' };

function resolve(m) {
  if (m === 'light' || m === 'dark') return m;
  return systemQuery?.matches ? 'dark' : 'light';
}

function apply(animate) {
  const resolved = resolve(mode);
  if (hasWindow) {
    const root = document.documentElement;
    if (animate) {
      root.classList.add('theme-anim'); // brief colour transition, only for a click on the toggle
      window.setTimeout(() => root.classList.remove('theme-anim'), 320);
    }
    root.dataset.theme = resolved;
    root.style.colorScheme = resolved;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#080d1c' : '#ffffff');
  }
  if (snapshot.mode !== mode || snapshot.resolved !== resolved) {
    snapshot = { mode, resolved };
    listeners.forEach((l) => l());
  }
}

export const subscribe = (l) => {
  listeners.add(l);
  return () => listeners.delete(l);
};
export const getSnapshot = () => snapshot;
/** The theme in force right now, "light" or "dark" (for canvas code that draws outside React). */
export const currentTheme = () => snapshot.resolved;

export function setMode(next) {
  if (!MODES.includes(next)) return;
  mode = next;
  try {
    window.localStorage.setItem(KEY, next);
  } catch {
    /* the choice just lasts for this visit */
  }
  apply(true);
}

if (hasWindow) {
  apply(false);
  const onSystem = () => mode === 'system' && apply(false);
  if (systemQuery?.addEventListener) systemQuery.addEventListener('change', onSystem);
  else systemQuery?.addListener?.(onSystem);
}

/** Tests only: reset the module state. */
export function __resetForTest(m = 'system') {
  mode = m;
  snapshot = { mode: m, resolved: resolve(m) };
  listeners.forEach((l) => l());
}
