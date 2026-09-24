import { describe, expect, it } from 'vitest';
import { cmap, gradientCss, STOPS } from './colormap.js';
import { checkNacaCode } from './naca.js';

describe('colormap', () => {
  it('returns the stop colours at the stop values', () => {
    for (const [v, rgb] of STOPS) expect(cmap(v)).toEqual(rgb);
  });
  it('clamps values outside [-2, 1]', () => {
    expect(cmap(-5)).toEqual(STOPS[0][1]);
    expect(cmap(9)).toEqual(STOPS[STOPS.length - 1][1]);
  });
  it('interpolates between stops', () => {
    const mid = cmap(-0.5);
    expect(mid[0]).toBeGreaterThan(79);
    expect(mid[0]).toBeLessThan(244);
  });
  it('builds a gradient with one entry per stop', () => {
    expect(gradientCss().match(/rgb\(/g)).toHaveLength(STOPS.length);
  });
});

describe('checkNacaCode', () => {
  it('accepts valid codes', () => {
    for (const c of ['0012', '2412', '2315']) expect(checkNacaCode(c)).toEqual({ valid: true, message: '' });
  });
  it('is silent for empty input', () => {
    expect(checkNacaCode('')).toEqual({ valid: false, message: '' });
  });
  it('explains invalid input', () => {
    expect(checkNacaCode('12').message).toBe('Enter exactly 4 digits.');
    expect(checkNacaCode('0003').message).toMatch(/at least 04/);
    expect(checkNacaCode('2015').message).toMatch(/must be 1-9/);
  });
});
