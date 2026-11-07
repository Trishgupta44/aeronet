// Helpers for the field object the API returns: { width, height, xmin, xmax, ymin, ymax, cp, u, v }.
// Values sit at cell centres, row-major, row j = 0 at the top (y = ymax). Cells inside the body or below the road are null.

/** World (x, y) -> grid coordinates (i, j); cell centres are at integer i and j. */
export function toGrid(f, x, y) {
  return [((x - f.xmin) / (f.xmax - f.xmin)) * f.width - 0.5, ((f.ymax - y) / (f.ymax - f.ymin)) * f.height - 0.5];
}

/** Grid coordinates (i, j) -> world (x, y). */
export function toWorld(f, i, j) {
  return [f.xmin + ((i + 0.5) / f.width) * (f.xmax - f.xmin), f.ymax - ((j + 0.5) / f.height) * (f.ymax - f.ymin)];
}

// Index of the lower-left cell of the bilinear stencil and the fractions, or null outside the grid of cell centres.
function stencil(f, x, y) {
  const gi = ((x - f.xmin) / (f.xmax - f.xmin)) * f.width - 0.5;
  const gj = ((f.ymax - y) / (f.ymax - f.ymin)) * f.height - 0.5;
  if (!(gi >= 0 && gi <= f.width - 1 && gj >= 0 && gj <= f.height - 1)) return null;
  const i0 = Math.min(Math.floor(gi), f.width - 2);
  const j0 = Math.min(Math.floor(gj), f.height - 2);
  return [j0 * f.width + i0, gi - i0, gj - j0];
}

function bilinear(a, w, k, fx, fy) {
  const v00 = a[k];
  const v10 = a[k + 1];
  const v01 = a[k + w];
  const v11 = a[k + w + 1];
  if (v00 === null || v10 === null || v01 === null || v11 === null) return null;
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
}

/** Bilinear sample of f[key] ('cp', 'u' or 'v') at a world point; null outside the grid or next to a null cell. */
export function sampleField(f, key, x, y) {
  const s = stencil(f, x, y);
  return s ? bilinear(f[key], f.width, s[0], s[1], s[2]) : null;
}

/** [u, v] at a world point (free-stream speed = 1), or null outside the grid or next to the body. */
export function sampleVelocity(f, x, y) {
  const s = stencil(f, x, y);
  if (!s) return null;
  const u = bilinear(f.u, f.width, s[0], s[1], s[2]);
  if (u === null) return null;
  const v = bilinear(f.v, f.width, s[0], s[1], s[2]);
  return v === null ? null : [u, v];
}
