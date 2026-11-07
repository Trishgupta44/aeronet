// Shareable page state in the query string. Freehand outlines are not stored; library shapes go by id.
// Unrelated flags (?present, ?light, ?sample) are left alone.

const TABS = ['explore', 'draw'];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const num = (s) => (s === null || s === '' || !Number.isFinite(+s) ? undefined : +s);

/** State in a query string -> a sanitised object. `alpha`, `mach` and `lifting` are undefined when absent. */
export function readState(search) {
  const q = new URLSearchParams(search);
  const tab = TABS.includes(q.get('tab')) ? q.get('tab') : 'explore';
  const code = /^\d{4}$/.test(q.get('code') ?? '') ? q.get('code') : '0012';
  const alpha = num(q.get('alpha'));
  const mach = num(q.get('mach'));
  const ground = num(q.get('ground'));
  const lifting = q.get('lifting');
  return {
    tab,
    code,
    alpha: alpha === undefined ? undefined : clamp(alpha, -10, 12),
    mach: mach === undefined ? undefined : clamp(mach, 0, 0.7),
    shape: (q.get('shape') ?? '').replace(/[^a-z0-9_-]/gi, '').slice(0, 24).toLowerCase(),
    lifting: lifting === null ? undefined : lifting === '1',
    ground: ground === undefined ? '' : clamp(ground, 0.02, 1),
  };
}

/** Query string (with the leading "?", or empty) that carries `s` on top of the unrelated parameters in `search`. */
export function writeState(search, s) {
  const q = new URLSearchParams(search);
  for (const k of ['tab', 'code', 'alpha', 'mach', 'shape', 'lifting', 'ground']) q.delete(k);
  q.set('tab', s.tab);
  q.set('code', s.code);
  if (s.alpha !== undefined) q.set('alpha', String(s.alpha));
  if (s.mach !== undefined) q.set('mach', String(s.mach));
  if (s.shape) q.set('shape', s.shape);
  if (s.lifting !== undefined) q.set('lifting', s.lifting ? '1' : '0');
  if (s.ground !== '' && s.ground !== undefined && s.ground !== null) q.set('ground', String(s.ground));
  const out = q.toString().replace(/=(?=&|$)/g, ''); // "present=" -> "present"
  return out ? `?${out}` : '';
}
