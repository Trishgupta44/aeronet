// Diverging colour map for Cp: suction (negative) blue, freestream neutral, stagnation red.
export const CP_MIN = -2;
export const CP_MAX = 1;
export const STOPS = [
  [-2, [26, 63, 156]],
  [-1, [79, 134, 217]],
  [0, [244, 243, 238]],
  [0.5, [240, 160, 90]],
  [1, [194, 59, 34]],
];

/** Map a Cp value to [r, g, b]; values outside [-2, 1] are clamped. */
export function cmap(v) {
  v = Math.max(CP_MIN, Math.min(CP_MAX, v));
  for (let i = 0; i + 1 < STOPS.length; i++) {
    if (v <= STOPS[i + 1][0]) {
      const f = (v - STOPS[i][0]) / (STOPS[i + 1][0] - STOPS[i][0]);
      const a = STOPS[i][1];
      const b = STOPS[i + 1][1];
      return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

/** CSS linear-gradient string matching the colour map (used for the colour bar). */
export function gradientCss() {
  const parts = STOPS.map((s, i) => `rgb(${s[1].join(',')}) ${(i / (STOPS.length - 1)) * 100}%`);
  return `linear-gradient(90deg,${parts.join(',')})`;
}
