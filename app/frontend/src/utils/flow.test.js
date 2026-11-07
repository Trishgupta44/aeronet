import { describe, expect, it } from 'vitest';
import { cmap, gradientCss, STOPS } from './colormap.js';
import { ISO_LEVELS, isoLines, isoSegments } from './contours.js';
import { sampleField, sampleVelocity, toGrid, toWorld } from './field.js';
import { createParticles, mulberry32, respawn, stepParticles } from './particles.js';

// 18 x 10 grid in the app window: cp = x at every cell centre, uniform flow u = 1, v = 0. `holes` lists [i, j] cells to null.
function fixture(holes = []) {
  const f = { width: 18, height: 10, xmin: -0.4, xmax: 1.4, ymin: -0.5, ymax: 0.5, cp: [], u: [], v: [] };
  for (let j = 0; j < f.height; j++) {
    for (let i = 0; i < f.width; i++) {
      const hole = holes.some(([a, b]) => a === i && b === j);
      f.cp.push(hole ? null : toWorld(f, i, j)[0]);
      f.u.push(hole ? null : 1);
      f.v.push(hole ? null : 0);
    }
  }
  return f;
}

describe('field sampling', () => {
  const f = fixture();
  it('toWorld and toGrid round-trip, with cell centres at integer indices', () => {
    const [x, y] = toWorld(f, 3, 4);
    const [i, j] = toGrid(f, x, y);
    expect(i).toBeCloseTo(3, 9);
    expect(j).toBeCloseTo(4, 9);
    expect(toWorld(f, 0, 0)[0]).toBeCloseTo(-0.4 + 0.05, 9); // half a cell in from the left edge
    expect(toWorld(f, 0, 0)[1]).toBeCloseTo(0.5 - 0.05, 9); // j = 0 is the top row
  });
  it('interpolates bilinearly (cp = x is reproduced exactly)', () => {
    expect(sampleField(f, 'cp', 0.333, 0.1)).toBeCloseTo(0.333, 9);
  });
  it('is null outside the grid and next to a null cell', () => {
    expect(sampleField(f, 'cp', -0.5, 0)).toBeNull();
    expect(sampleField(f, 'cp', 1.5, 0)).toBeNull();
    expect(sampleField(f, 'cp', 0, 0.6)).toBeNull();
    const [x, y] = toWorld(fixture(), 8, 5);
    const g = fixture([[8, 5]]);
    expect(sampleField(g, 'cp', x + 0.02, y + 0.02)).toBeNull(); // one of its 4 neighbours is the hole
  });
  it('gives the free-stream velocity in a free area', () => {
    expect(sampleVelocity(f, 0.5, 0.0)).toEqual([1, 0]);
    expect(sampleVelocity(fixture([[8, 5]]), toWorld(f, 8, 5)[0], toWorld(f, 8, 5)[1])).toBeNull();
  });
});

describe('particles', () => {
  const f = fixture();
  it('moves one RK2 step in uniform flow by exactly dt', () => {
    const p = { x: 0.2, y: 0.1, px: 0.2, py: 0.1, age: 0, life: 100 };
    stepParticles([p], f, 0.01, mulberry32(1));
    expect(p.x).toBeCloseTo(0.21, 12);
    expect(p.y).toBeCloseTo(0.1, 12);
    expect(p.px).toBe(0.2);
    expect(p.age).toBe(1);
  });
  it('respawns on the inflow edge when it leaves the window', () => {
    const p = { x: 1.3, y: 0, px: 1.3, py: 0, age: 40, life: 100 };
    stepParticles([p], f, 0.5, mulberry32(3));
    expect(p.x).toBeLessThan(-0.3);
    expect(p.age).toBe(0);
    expect(p.px).toBe(p.x); // no streak across the window
  });
  it('respawn(fromLeft) lands in the left 4 % of the window, in a free cell', () => {
    const rand = mulberry32(11);
    for (let k = 0; k < 50; k++) {
      const p = { x: 0, y: 0, px: 0, py: 0, age: 9, life: 9 };
      respawn(p, f, rand, true);
      expect(p.x).toBeLessThan(-0.4 + 0.04 * 1.8 + 1e-9);
      expect(sampleVelocity(f, p.x, p.y)).not.toBeNull();
      expect(p.life).toBeGreaterThanOrEqual(80);
      expect(p.life).toBeLessThanOrEqual(200);
    }
  });
  it('createParticles is deterministic for a seed and starts every particle in a free cell', () => {
    const g = fixture([[8, 5], [9, 5], [8, 4], [9, 4]]);
    const a = createParticles(50, g, mulberry32(7));
    const b = createParticles(50, g, mulberry32(7));
    expect(a).toEqual(b);
    expect(a).toHaveLength(50);
    for (const p of a) expect(sampleVelocity(g, p.x, p.y)).not.toBeNull();
  });
});

describe('isobars', () => {
  it('uses the planned levels (no zero line)', () => {
    expect(ISO_LEVELS[0]).toBe(-1.5);
    expect(ISO_LEVELS[ISO_LEVELS.length - 1]).toBe(0.75);
    expect(ISO_LEVELS).not.toContain(0);
    expect(ISO_LEVELS).toHaveLength(9);
  });
  it('finds the vertical line cp = 0.5 in the cp = x field', () => {
    const segs = isoSegments(fixture(), 0.5);
    expect(segs.length).toBeGreaterThan(5);
    for (const [x1, , x2] of segs) {
      expect(x1).toBeCloseTo(0.5, 9);
      expect(x2).toBeCloseTo(0.5, 9);
    }
  });
  it('gives fewer segments when a cell is null', () => {
    const full = isoSegments(fixture(), 0.5).length;
    const holey = isoSegments(fixture([[9, 5]]), 0.5).length; // cell 9 sits at x = 0.5 .. 0.55
    expect(holey).toBeLessThan(full);
  });
  it('chains the segments into one polyline per line', () => {
    const lines = isoLines(fixture(), 0.5);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toHaveLength(10); // 10 grid rows
  });
  it('finds nothing when the level is outside the data', () => {
    expect(isoSegments(fixture(), 5)).toEqual([]);
  });
});

describe('colour map (new stops)', () => {
  it('has the eight planned stops from -2 to 1', () => {
    expect(STOPS).toHaveLength(8);
    expect(STOPS[0][0]).toBe(-2);
    expect(STOPS[STOPS.length - 1][0]).toBe(1);
    expect(cmap(0)).toEqual([247, 247, 244]);
  });
  it('places gradient stops at their Cp position, not evenly', () => {
    const css = gradientCss();
    expect(css).toContain('rgb(247,247,244) 66.66');
    expect(css.match(/rgb\(/g)).toHaveLength(8);
  });
});
