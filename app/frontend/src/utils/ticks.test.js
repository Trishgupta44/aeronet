import { describe, expect, it } from 'vitest';
import { decadeLabel, decadeTicks, niceTicks } from './ticks.js';

describe('niceTicks', () => {
  it('picks round steps inside the range', () => {
    expect(niceTicks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(niceTicks(0.28, 0.52, 4)).toEqual([0.3, 0.4, 0.5]);
    expect(niceTicks(0, 20, 5)).toEqual([0, 5, 10, 15, 20]);
  });
  it('returns the single value for an empty range', () => {
    expect(niceTicks(2, 2)).toEqual([2]);
  });
});

describe('decade ticks', () => {
  it('labels small decades as plain numbers and large ones in e-notation', () => {
    expect(decadeLabel(-2)).toBe('0.01');
    expect(decadeLabel(0)).toBe('1');
    expect(decadeLabel(17)).toBe('1e17');
    expect(decadeLabel(-6)).toBe('1e-6');
  });
  it('thins the decades so at most `max` labels are drawn', () => {
    expect(decadeTicks(-3, 0, 5)).toEqual([-3, -2, -1, 0]);
    expect(decadeTicks(9, 18, 5)).toEqual([9, 11, 13, 15, 17]);
  });
});
