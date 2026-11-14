import { describe, expect, it } from 'vitest';
import { keyPoints } from './CpCard.jsx';

// A small made-up surface: only the arithmetic of the three annotations is tested here.
const surface = {
  cx: [0.0, 0.05, 0.5, 1.0, 0.0, 0.5, 0.98],
  cp: [1.0, -1.6, -0.4, 0.2, 0.9, 0.1, 0.0],
  upper: [true, true, true, true, false, false, false],
};

describe('keyPoints', () => {
  it('finds the stagnation point as the highest Cp', () => {
    expect(keyPoints(surface).stagnation).toMatchObject({ x: 0.0, cp: 1.0 });
  });
  it('finds the suction peak as the lowest Cp and reports its x', () => {
    expect(keyPoints(surface).peak).toMatchObject({ x: 0.05, cp: -1.6 });
  });
  it('averages the two surface points with the largest x for the trailing edge', () => {
    const te = keyPoints(surface).trailing;
    expect(te.x).toBe(1.0);
    expect(te.cp).toBeCloseTo((0.2 + 0.0) / 2, 12);
  });
  it('copes with an empty surface', () => {
    expect(keyPoints({ cx: [], cp: [], upper: [] })).toEqual({});
  });
});
