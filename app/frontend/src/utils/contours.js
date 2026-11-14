import { toWorld } from './field.js';

/** Isobar levels: Cp from -1.5 to 0.75 in steps of 0.25, without the zero line. */
export const ISO_LEVELS = [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75];

// Corner case (tl=8, tr=4, br=2, bl=1) -> pairs of cell edges joined by a line. Edges: T(op), R(ight), B(ottom), L(eft).
const CASES = [
  [], ['LB'], ['BR'], ['LR'], ['TR'], null /* 5: saddle */, ['TB'], ['LT'],
  ['LT'], ['TB'], null /* 10: saddle */, ['TR'], ['LR'], ['BR'], ['LB'], [],
];

/**
 * Marching squares on the cell-centre grid. Returns line segments [x1, y1, x2, y2] in world coordinates where
 * f[key] crosses `level`. Cells with a null corner are skipped, so no line is drawn into the body.
 */
export function isoSegments(f, level, key = 'cp') {
  const a = f[key];
  const w = f.width;
  const out = [];
  for (let j = 0; j < f.height - 1; j++) {
    for (let i = 0; i < w - 1; i++) {
      const k = j * w + i;
      const tl = a[k];
      const tr = a[k + 1];
      const bl = a[k + w];
      const br = a[k + w + 1];
      if (tl === null || tr === null || bl === null || br === null) continue;
      const idx = (tl >= level ? 8 : 0) | (tr >= level ? 4 : 0) | (br >= level ? 2 : 0) | (bl >= level ? 1 : 0);
      if (idx === 0 || idx === 15) continue;
      // crossing points, always interpolated from the lower-index corner so shared edges agree exactly
      const edge = {
        T: () => toWorld(f, i + (level - tl) / (tr - tl), j),
        R: () => toWorld(f, i + 1, j + (level - tr) / (br - tr)),
        B: () => toWorld(f, i + (level - bl) / (br - bl), j + 1),
        L: () => toWorld(f, i, j + (level - tl) / (bl - tl)),
      };
      let pairs = CASES[idx];
      if (pairs === null) {
        const centreHigh = (tl + tr + bl + br) / 4 >= level;
        if (idx === 5) pairs = centreHigh ? ['LT', 'BR'] : ['TR', 'LB']; // tr and bl high
        else pairs = centreHigh ? ['TR', 'LB'] : ['LT', 'BR']; // tl and br high
      }
      for (const pair of pairs) {
        const p = edge[pair[0]]();
        const q = edge[pair[1]]();
        out.push([p[0], p[1], q[0], q[1]]);
      }
    }
  }
  return out;
}

/** Join the segments of one level into polylines [[x, y], ...] (so dashed lines dash along the whole line). */
export function chainSegments(segs) {
  const key = (x, y) => `${Math.round(x * 1e6)},${Math.round(y * 1e6)}`;
  const at = new Map();
  segs.forEach((s, n) => {
    for (const e of [key(s[0], s[1]), key(s[2], s[3])]) {
      if (!at.has(e)) at.set(e, []);
      at.get(e).push(n);
    }
  });
  const used = new Uint8Array(segs.length);
  const extend = (line) => {
    for (;;) {
      const last = line[line.length - 1];
      const next = (at.get(key(last[0], last[1])) || []).find((n) => !used[n]);
      if (next === undefined) return;
      used[next] = 1;
      const s = segs[next];
      const startsHere = key(s[0], s[1]) === key(last[0], last[1]);
      line.push(startsHere ? [s[2], s[3]] : [s[0], s[1]]);
    }
  };
  const lines = [];
  segs.forEach((s, n) => {
    if (used[n]) return;
    used[n] = 1;
    const line = [[s[0], s[1]], [s[2], s[3]]];
    extend(line);
    line.reverse();
    extend(line);
    lines.push(line);
  });
  return lines;
}

export const isoLines = (f, level, key = 'cp') => chainSegments(isoSegments(f, level, key));
