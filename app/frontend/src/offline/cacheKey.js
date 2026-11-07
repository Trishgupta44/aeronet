// Key of a saved API answer. Shared by the browser (lookup) and scripts/make-cache.mjs (writing), so both always agree.
// Pure JavaScript: no DOM, no React.
//
// Only the inputs that change the answer are part of the key. `grid_width` is not (the saved field has one resolution), and a
// reference overlay counts only as "present, with N points" (the saved case is the bundled sample).

const num = (v) => (v == null ? 'n' : String(Math.round(v * 1000) / 1000));
const refPart = (body) => `ref${body.reference ? body.reference.length : 0}`;

/** 32-bit FNV-1a hash of an outline, rounded to 4 decimals, as 8 hex digits. */
export function hashPoints(points) {
  let h = 0x811c9dc5;
  const s = points.map((p) => `${p[0].toFixed(4)},${p[1].toFixed(4)}`).join(';');
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export function cacheKey(path, body) {
  if (!body) return path;
  switch (path) {
    case '/api/solve':
      return `${path}|${body.solver ?? 'panel'}|${body.code}|${num(body.alpha)}|${num(body.mach)}|${refPart(body)}`;
    case '/api/solve_shape':
      return `${path}|${hashPoints(body.points)}|${num(body.alpha)}|${num(body.mach)}|${body.lifting ? 1 : 0}|${num(body.ground)}|${refPart(body)}`;
    case '/api/polar': {
      const who = body.code ? `code${body.code}` : `pts${hashPoints(body.points)}|${body.lifting === false ? 0 : 1}|${num(body.ground)}`;
      const a = body.alphas;
      return `${path}|${who}|${num(body.mach)}|${a.length}:${num(a[0])}:${num(a[a.length - 1])}`;
    }
    case '/api/viscous':
      return `${path}|${body.code}|${num(body.alpha)}|${num(body.re)}`;
    default:
      return `${path}|${JSON.stringify(body)}`;
  }
}
