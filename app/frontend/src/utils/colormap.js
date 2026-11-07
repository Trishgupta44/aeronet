// Perceptual diverging colour map for Cp (RdBu-like): suction (negative) blue, freestream near-white, stagnation red.
export const CP_MIN = -2;
export const CP_MAX = 1;
export const STOPS = [
  [-2, [23, 37, 99]],
  [-1.4, [33, 84, 160]],
  [-0.8, [67, 147, 195]],
  [-0.3, [166, 206, 227]],
  [0, [247, 247, 244]],
  [0.35, [253, 204, 138]],
  [0.7, [227, 105, 70]],
  [1, [165, 15, 38]],
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
  const parts = STOPS.map((s) => `rgb(${s[1].join(',')}) ${((s[0] - CP_MIN) / (CP_MAX - CP_MIN)) * 100}%`);
  return `linear-gradient(90deg,${parts.join(',')})`;
}
