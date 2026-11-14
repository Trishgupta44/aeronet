// Axis tick helpers for the small canvas charts. Pure functions.

/** About `count` round tick values covering [lo, hi] (steps of 1, 2 or 5 times a power of ten). The ends may lie outside [lo, hi]. */
export function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) out.push(Number((k * step).toPrecision(12)));
  return out;
}

/** Labels for decades of a log axis: 0.001 ... 1000 as plain numbers, anything else as 1e-5, 1e17. */
export function decadeLabel(k) {
  return k >= -3 && k <= 3 ? String(10 ** k) : `1e${k}`;
}

/** Integer exponents k with 10^k inside [10^kLo, 10^kHi] spaced so that at most `max` labels are shown. */
export function decadeTicks(kLo, kHi, max = 5) {
  const step = Math.max(1, Math.ceil((kHi - kLo + 1) / max));
  const out = [];
  for (let k = Math.ceil(kLo); k <= kHi; k += step) out.push(k);
  return out;
}

/** Ticks at 1, 2 and 5 of every decade between 10^kLo and 10^kHi (exponents): `[value, label]`. For log axes that span only one or two decades. */
export function logTicks125(kLo, kHi) {
  const out = [];
  for (let k = Math.floor(kLo); k <= Math.ceil(kHi); k++) {
    for (const m of [1, 2, 5]) {
      const v = Number((m * 10 ** k).toPrecision(6));
      const lv = Math.log10(v);
      if (lv >= kLo - 1e-9 && lv <= kHi + 1e-9) out.push([v, k >= -3 && k <= 3 ? String(v) : v.toExponential(0)]);
    }
  }
  return out;
}
