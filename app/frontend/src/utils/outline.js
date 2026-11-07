// Helpers for drawn outlines: smoothing, thinning, mirroring, file formats. Points are [x, y] pairs in chords.

/** Corner-cutting smoothing of a closed polygon; every pass doubles the number of points. */
export function chaikin(points, iters = 2) {
  let p = points.map((q) => [q[0], q[1]]);
  for (let it = 0; it < iters; it++) {
    const out = [];
    for (let i = 0; i < p.length; i++) {
      const a = p[i];
      const b = p[(i + 1) % p.length];
      out.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
      out.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    p = out;
  }
  return p;
}

/** Keep the first point, then every point at least `minDist` away from the last one kept. */
export function thin(points, minDist) {
  const out = [];
  for (const q of points) {
    const last = out[out.length - 1];
    if (!last || Math.hypot(q[0] - last[0], q[1] - last[1]) >= minDist) out.push([q[0], q[1]]);
  }
  return out;
}

/** An upper-half stroke (drawn left to right or back) becomes a closed outline: the stroke, then its mirror image about y = 0, reversed. */
export function mirrorClosed(stroke) {
  return stroke.map((q) => [q[0], q[1]]).concat(stroke.map((q) => [q[0], -q[1]]).reverse());
}

/**
 * The outline is always treated as closed. Drops trailing points that lie within `snap` of the start (so the join is not
 * sampled twice) and reports how large the gap was before closing: `gap` is the distance from the last point to the first.
 */
export function autoClose(points, snap = 0.02) {
  const pts = points.map((q) => [q[0], q[1]]);
  const first = pts[0];
  const gap = first ? Math.hypot(pts[pts.length - 1][0] - first[0], pts[pts.length - 1][1] - first[1]) : 0;
  while (pts.length > 3 && Math.hypot(pts[pts.length - 1][0] - first[0], pts[pts.length - 1][1] - first[1]) < snap) pts.pop();
  return { points: pts, gap };
}

const MAX_POINTS = 900; // keeps the request (after smoothing: 4x) well below the API limit of 4000

/**
 * From a raw pointer stroke to the outline we send: thin, check the size, fold (mirror mode), close, smooth.
 * Returns `{ points, closedGap }` or `{ error }`.
 */
export function processStroke(stroke, { mirror = false } = {}) {
  let pts = thin(stroke, 0.004);
  for (let d = 0.006; pts.length > MAX_POINTS; d *= 1.5) pts = thin(pts, d);
  if (pts.length < 8) return { error: 'Draw a bigger loop.' };
  let gap = 0;
  if (mirror) {
    pts = mirrorClosed(pts.map((q) => [q[0], Math.max(0, q[1])]));
  } else {
    const closed = autoClose(pts);
    pts = closed.points;
    gap = closed.gap;
  }
  if (pts.length < 8) return { error: 'Draw a bigger loop.' };
  return { points: chaikin(pts, 2), closedGap: gap };
}

const fix6 = (v) => v.toFixed(6);

/** Selig-format `.dat` text: a name line, then the outline as `x y` rows, closed by repeating the first point. */
export function toDat(points, name = 'AeroNet shape') {
  const rows = points.concat(points.length ? [points[0]] : []);
  return `${name}\n${rows.map((q) => `  ${fix6(q[0])}  ${fix6(q[1])}`).join('\n')}\n`;
}

/** Surface pressure table of a solution: `x,y,cp,upper` with 6 decimals. */
export function surfaceCsv(result) {
  const s = result.surface;
  const rows = s.cx.map((x, i) => `${fix6(x)},${fix6(s.cy[i])},${fix6(s.cp[i])},${s.upper[i] ? 1 : 0}\n`);
  return `x,y,cp,upper\n${rows.join('')}`;
}
