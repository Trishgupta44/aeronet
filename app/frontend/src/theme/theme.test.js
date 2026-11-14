import { beforeEach, describe, expect, it, vi } from 'vitest';
import { cmap, STOPS, STOPS_DARK, stopsFor } from '../utils/colormap.js';
import { pageFromHash, PAGES } from '../utils/route.js';
import { canvasPalette, CANVAS } from './canvasPalette.js';
import { __resetForTest, currentTheme, getSnapshot, setMode, subscribe } from './theme.js';

const luma = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const hex = (rgb) => `#${rgb.map((v) => v.toString(16).padStart(2, '0')).join('')}`;

describe('theme store', () => {
  beforeEach(() => __resetForTest('system'));

  it('resolves system to light when the environment has no dark preference', () => {
    expect(getSnapshot()).toEqual({ mode: 'system', resolved: 'light' });
  });

  it('switches between light and dark, tells subscribers, and ignores unknown modes', () => {
    const seen = vi.fn();
    const off = subscribe(seen);
    setMode('dark');
    expect(getSnapshot()).toEqual({ mode: 'dark', resolved: 'dark' });
    expect(currentTheme()).toBe('dark');
    setMode('sepia');
    expect(getSnapshot().mode).toBe('dark');
    setMode('light');
    expect(currentTheme()).toBe('light');
    expect(seen).toHaveBeenCalledTimes(2);
    off();
    setMode('dark');
    expect(seen).toHaveBeenCalledTimes(2);
  });
});

describe('dark colour map', () => {
  it('has the same Cp stops as the light map, so the colour bar labels stay valid', () => {
    expect(STOPS_DARK.map((s) => s[0])).toEqual(STOPS.map((s) => s[0]));
    expect(stopsFor('dark')).toBe(STOPS_DARK);
    expect(stopsFor('light')).toBe(STOPS);
  });
  it('is the page navy at the free-stream value, equal to the dark plot stage colour', () => {
    expect(hex(cmap(0, STOPS_DARK))).toBe(CANVAS.dark.stage);
    expect(hex(cmap(0))).toBe(CANVAS.light.stage); // and the light map's neutral is the light stage
  });
  it('gets brighter with the size of the pressure change on both sides (strong suction and stagnation glow)', () => {
    const suction = [0, -0.3, -0.8, -1.4, -2].map((v) => luma(cmap(v, STOPS_DARK)));
    const pressure = [0, 0.35, 0.7, 1].map((v) => luma(cmap(v, STOPS_DARK)));
    for (const run of [suction, pressure]) for (let i = 1; i < run.length; i += 1) expect(run[i]).toBeGreaterThan(run[i - 1]);
  });
  it('gives each theme its own canvas palette', () => {
    expect(canvasPalette('dark')).toBe(CANVAS.dark);
    expect(canvasPalette('light')).toBe(CANVAS.light);
    expect(CANVAS.dark.bodyFill).not.toBe(CANVAS.light.bodyFill);
  });
});

describe('pages', () => {
  it('reads the page from the hash, and nothing for other anchors', () => {
    for (const p of PAGES) expect(pageFromHash(`#${p}`)).toBe(p);
    expect(pageFromHash('#credit')).toBeNull();
    expect(pageFromHash('')).toBeNull();
    expect(pageFromHash(undefined)).toBeNull();
  });
});
