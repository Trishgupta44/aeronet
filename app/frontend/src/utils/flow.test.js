import { describe, expect, it } from 'vitest';
import { cmap, gradientCss, STOPS } from './colormap.js';
import { ISO_LEVELS, isoLines, isoSegments } from './contours.js';
import { sampleField, sampleVelocity, toGrid, toWorld } from './field.js';
import { createParticles, mulberry32, respawn, stepParticles } from './particles.js';
import { autoClose, chaikin, mirrorClosed, processStroke, surfaceCsv, thin, toDat } from './outline.js';
import { readState, writeState } from './urlState.js';

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

describe('outline helpers', () => {
  it('chaikin on a square gives 16 points (two passes, closed polygon)', () => {
    const sq = [[0, 0], [1, 0], [1, 1], [0, 1]];
    const out = chaikin(sq);
    expect(out).toHaveLength(16);
    expect(out[0][0]).toBeCloseTo(0.375, 9); // first pass gives (0.25, 0) and (0.75, 0); the second cuts between them
  });
  it('chaikin with zero passes returns a copy', () => {
    const sq = [[0, 0], [1, 0], [1, 1]];
    expect(chaikin(sq, 0)).toEqual(sq);
  });
  it('thin drops points closer than minDist to the last kept one', () => {
    expect(thin([[0, 0], [0.001, 0], [0.1, 0]], 0.01)).toEqual([[0, 0], [0.1, 0]]);
    expect(thin([], 0.01)).toEqual([]);
  });
  it('mirrorClosed reflects the stroke about y = 0 and walks back', () => {
    expect(mirrorClosed([[0, 0.1], [1, 0.2]])).toEqual([[0, 0.1], [1, 0.2], [1, -0.2], [0, -0.1]]);
  });
  it('autoClose drops trailing points that sit on the start, and reports the gap', () => {
    const r = autoClose([[0, 0], [1, 0], [1, 1], [0.001, 0.001]], 0.02);
    expect(r.points).toEqual([[0, 0], [1, 0], [1, 1]]);
    expect(r.gap).toBeCloseTo(Math.hypot(0.001, 0.001), 9);
    const open = autoClose([[0, 0], [1, 0], [1, 1]], 0.02);
    expect(open.points).toHaveLength(3);
    expect(open.gap).toBeCloseTo(Math.SQRT2, 9);
  });
  it('processStroke rejects a tiny scribble and accepts a loop (mirror mode folds the stroke into the upper half)', () => {
    const tiny = processStroke([[0, 0], [0.01, 0.01], [0.02, 0]], { mirror: false });
    expect(tiny.error).toMatch(/bigger loop/i);
    const circle = Array.from({ length: 60 }, (_, k) => [0.5 + 0.3 * Math.cos((k / 60) * 6.2832), 0.2 * Math.sin((k / 60) * 6.2832)]);
    const ok = processStroke(circle, { mirror: false });
    expect(ok.error).toBeUndefined();
    expect(ok.points.length).toBeGreaterThanOrEqual(8);
    const half = Array.from({ length: 30 }, (_, k) => [(k / 29) * 0.9 + 0.05, 0.12 * Math.sin((k / 29) * Math.PI) - 0.02]); // dips below 0 at the ends
    const m = processStroke(half, { mirror: true });
    expect(m.error).toBeUndefined();
    const ys = m.points.map((p) => p[1]);
    expect(Math.max(...ys)).toBeGreaterThan(0.05);
    expect(Math.min(...ys)).toBeLessThan(-0.05);
  });
  it('toDat writes a name line plus n + 1 rows (the outline is closed)', () => {
    const lines = toDat([[1, 0], [0.5, 0.1], [0, 0], [0.5, -0.1]], 'blob').trim().split('\n');
    expect(lines[0]).toBe('blob');
    expect(lines).toHaveLength(1 + 4 + 1);
    expect(lines[1]).toBe(lines[lines.length - 1]);
    expect(lines[2].trim().split(/\s+/).map(Number)).toEqual([0.5, 0.1]);
  });
  it('surfaceCsv writes x, y, cp and an upper flag', () => {
    const r = { surface: { cx: [0.5], cy: [0.1], cp: [-0.3], upper: [true] } };
    expect(surfaceCsv(r)).toBe('x,y,cp,upper\n0.500000,0.100000,-0.300000,1\n');
  });
});

describe('url state', () => {
  it('round-trips the explore state and keeps unrelated flags', () => {
    const s = { tab: 'explore', code: '2412', alpha: 6, mach: 0.3, shape: '', lifting: true, ground: '' };
    const search = writeState('?present', s);
    expect(new URLSearchParams(search).has('present')).toBe(true);
    expect(readState(search)).toEqual(s);
  });
  it('round-trips the draw state (library shapes are stored by id)', () => {
    const s = { tab: 'draw', code: '0012', alpha: 0, mach: 0, shape: 'sedan', lifting: false, ground: 0.03 };
    expect(readState(writeState('', s))).toEqual(s);
  });
  it('sanitises hostile or out-of-range input', () => {
    expect(readState('?tab=evil&code=12&alpha=99&mach=-1&shape=<x>&ground=9')).toEqual({ tab: 'explore', code: '0012', alpha: 12, mach: 0, shape: 'x', ground: 1 });
  });
  it('replaces earlier values instead of appending', () => {
    const once = writeState('?light&alpha=3', { tab: 'explore', code: '0012', alpha: 5, mach: 0.4, shape: '', ground: '' });
    expect(new URLSearchParams(once).getAll('alpha')).toEqual(['5']);
    expect(new URLSearchParams(once).has('light')).toBe(true);
  });
});
