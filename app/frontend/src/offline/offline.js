// Safe demo mode: when the solver backend stops answering, saved answers (public/cache/bundle.json, written by
// scripts/make-cache.mjs) are served instead, and a banner says so. The store has no React in it so api.js can use it.

import { cacheKey } from './cacheKey.js';

const BUNDLE_URL = '/cache/bundle.json';
const POLL_MS = 10000;
const HEALTH_TIMEOUT_MS = 3000;

// Static demo build (VITE_STATIC_DEMO=1): there is no backend at all (the page is hosted as plain files), so the saved answers
// are used from the first request on and nothing calls /api. Everything else is the same as the offline fallback.
export const STATIC_DEMO = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_STATIC_DEMO === '1') || false;

const store = { offline: STATIC_DEMO, bundle: null, loading: null, retryFailed: false };
let snapshot = { ...store };
const listeners = new Set();

function publish() {
  snapshot = { offline: store.offline, savedAt: store.bundle?.generated_utc ?? null, hasBundle: !!store.bundle, retryFailed: store.retryFailed, staticDemo: STATIC_DEMO };
  listeners.forEach((l) => l());
}
publish();

export const subscribe = (l) => {
  listeners.add(l);
  return () => listeners.delete(l);
};
export const getSnapshot = () => snapshot;
export const isOffline = () => store.offline;

/** Load the saved answers once (in the background while the backend is up, so they exist when it dies). */
export function ensureBundle() {
  if (!store.loading) {
    store.loading = fetch(BUNDLE_URL, { cache: 'no-cache' })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .then((b) => {
        store.bundle = b && b.entries ? b : null;
        publish();
        return store.bundle;
      });
  }
  return store.loading;
}

export function goOffline() {
  if (store.offline) return;
  store.offline = true;
  ensureBundle();
  publish();
}

/** The saved answer for a request, or an Error whose message starts with "Cannot reach" (the pad keeps the shape on such errors). */
export async function offlineLookup(path, body) {
  const bundle = await ensureBundle();
  if (!bundle) throw new Error('Cannot reach the solver API.');
  const hit = bundle.entries[cacheKey(path, body)];
  if (hit === undefined) {
    throw new Error('Cannot reach the solver API, and this case was not saved. Offline, only the presets at their starting angle and Mach number are available.');
  }
  return hit;
}

async function healthy() {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), HEALTH_TIMEOUT_MS);
  try {
    const r = await fetch('/api/health', { cache: 'no-store', signal: ctl.signal });
    return r.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

/** Poll the health endpoint while online; the first failure switches to saved results and polling stops. */
export function startHealthMonitor() {
  ensureBundle();
  if (STATIC_DEMO) return () => {}; // no backend to watch
  const id = setInterval(async () => {
    if (store.offline) return;
    if (!(await healthy())) goOffline();
  }, POLL_MS);
  return () => clearInterval(id);
}

/** "Try again": reload the page when the backend answers again (every card then fetches live data). */
export async function retry() {
  if (await healthy()) {
    window.location.reload();
    return;
  }
  store.retryFailed = true;
  publish();
}

/** Tests only. */
export function __setForTest({ bundle, offline }) {
  store.bundle = bundle ?? null;
  store.loading = Promise.resolve(store.bundle);
  store.offline = !!offline;
  publish();
}
