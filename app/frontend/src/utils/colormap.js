// Perceptual diverging colour map for Cp: suction (negative) blue, free stream neutral, stagnation red.
// Two versions: STOPS (light theme, RdBu-like, neutral = near-white) and STOPS_DARK (dark theme, neutral = the page's deep navy,
// brightness rising with the size of the pressure change on both sides, so the strong suction and the stagnation point glow).
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
export const STOPS_DARK = [
  [-2, [196, 235, 255]],
  [-1.4, [104, 188, 244]],
  [-0.8, [48, 124, 210]],
  [-0.3, [26, 66, 136]],
  [0, [12, 20, 48]],
  [0.35, [92, 40, 62]],
  [0.7, [226, 98, 62]],
  [1, [255, 208, 128]],
];

/** The stops for a theme ("light" or "dark"). */
export const stopsFor = (theme) => (theme === 'dark' ? STOPS_DARK : STOPS);

/** Map a Cp value to [r, g, b]; values outside [-2, 1] are clamped. `stops` defaults to the light map. */
export function cmap(v, stops = STOPS) {
  v = Math.max(CP_MIN, Math.min(CP_MAX, v));
  for (let i = 0; i + 1 < stops.length; i++) {
    if (v <= stops[i + 1][0]) {
      const f = (v - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      const a = stops[i][1];
      const b = stops[i + 1][1];
      return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
    }
  }
  return stops[stops.length - 1][1];
}

/** CSS linear-gradient string matching the colour map (used for the colour bar). */
export function gradientCss(stops = STOPS) {
  const parts = stops.map((s) => `rgb(${s[1].join(',')}) ${((s[0] - CP_MIN) / (CP_MAX - CP_MIN)) * 100}%`);
  return `linear-gradient(90deg,${parts.join(',')})`;
}
