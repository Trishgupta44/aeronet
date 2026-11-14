import { describe, expect, it } from 'vitest';
import { fmt, fmtMs, fmtSec, meshLabel } from './format.js';

describe('format', () => {
  it('fmt rounds, hides negative zero and shows an en dash for missing values', () => {
    expect(fmt(0.07340755, 4)).toBe('0.0734');
    expect(fmt(-0.0000001, 3)).toBe('0.000');
    expect(fmt(null)).toBe('–');
    expect(fmt(undefined)).toBe('–');
  });
  it('fmtMs picks a readable unit', () => {
    expect(fmtMs(1.2651)).toBe('1.27 ms');
    expect(fmtMs(16.7665)).toBe('16.8 ms');
    expect(fmtMs(150.4)).toBe('150 ms');
    expect(fmtMs(9000)).toBe('9 s');
    expect(fmtMs(285000)).toBe('285 s');
    expect(fmtMs(null)).toBe('–');
  });
  it('fmtSec and meshLabel', () => {
    expect(fmtSec(334.2)).toBe('334.2 s');
    expect(fmtSec(null)).toBe('–');
    expect(meshLabel('50x25')).toBe('50×25');
    expect(meshLabel('140 panels')).toBe('140 panels');
    expect(meshLabel(null)).toBe('–');
  });
});
