// Number formatting for the results tables and tiles. A missing value is an en dash; nothing is ever filled in.

const DASH = '–';

/** `v` with `d` decimals (no "-0.000"); an en dash for null or NaN. */
export function fmt(v, d = 3) {
  if (v == null || Number.isNaN(v)) return DASH;
  return (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d);
}

/** A time given in milliseconds: "1.27 ms", "24.5 ms", "285 s" (from 1000 ms up). */
export function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return DASH;
  if (ms >= 1000) return `${Number((ms / 1000).toPrecision(4))} s`;
  if (ms >= 100) return `${ms.toFixed(0)} ms`;
  if (ms >= 10) return `${ms.toFixed(1)} ms`;
  return `${ms.toFixed(2)} ms`;
}

/** A time given in seconds: "334.2 s". */
export const fmtSec = (s) => (s == null || Number.isNaN(s) ? DASH : `${s.toFixed(1)} s`);

/** "50x25" -> "50×25". */
export const meshLabel = (mesh) => (mesh ? String(mesh).replace('x', '×') : DASH);
