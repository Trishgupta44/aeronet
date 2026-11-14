import { describe, expect, it } from 'vitest';
import { activeSection } from './scrollSpy.js';

const offsets = [
  { id: 'explore', top: 700 },
  { id: 'draw', top: 1700 },
  { id: 'results', top: 2900 },
];

describe('activeSection', () => {
  it('is null while still above the first section', () => {
    expect(activeSection(offsets, 0)).toBeNull();
    expect(activeSection(offsets, 600)).toBeNull(); // 600 + 80 < 700
  });
  it('picks the last section whose top is above scrollY + 80', () => {
    expect(activeSection(offsets, 650)).toBe('explore'); // 730 >= 700
    expect(activeSection(offsets, 1650)).toBe('draw');
  });
  it('stays on the last section at the bottom of the page', () => {
    expect(activeSection(offsets, 5000)).toBe('results');
  });
  it('does not depend on the order of the offsets', () => {
    expect(activeSection([...offsets].reverse(), 1650)).toBe('draw');
  });
  it('returns a non-nav section (the cards band) once the reader is in it, so the nav can highlight nothing', () => {
    const withBand = [...offsets, { id: 'overview', top: 2300 }];
    expect(activeSection(withBand, 2250)).toBe('overview');
    expect(activeSection(withBand, 2900)).toBe('results');
  });
});
