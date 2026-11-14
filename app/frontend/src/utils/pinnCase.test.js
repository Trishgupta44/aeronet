import { describe, expect, it } from 'vitest';
import { yRange } from '../components/PinnCpChart.jsx';
import { caseGroups, caseToResult, cpError, findRow, fvmView, methodLabel, replayCurve, roundRms, rowsFor, shortVariant, snapshotResult } from './pinnCase.js';

// A 6-face toy case: the faces run counter-clockwise from the trailing edge, so the minimum x (the leading edge) is at face 3.
// Reference points lie exactly on the faces, so the Cp error of the final round is easy to state by hand.
const TOY_CP = [0.1, -0.2, -0.5, 1.0, -0.3, 0.0];
function toy(extra = {}) {
  const sx = [0.9, 0.5, 0.1, 0.0, 0.2, 0.7];
  const sy = [0.02, 0.05, 0.04, 0.0, -0.04, -0.03];
  return {
    id: 'toy',
    label: 'Toy · 6×3 · 4° · Ma 0.4 · positivity on',
    shape: 'Toy',
    alpha: 4,
    mach: 0.4,
    x: [1, 0.8, 0.3, 0, 0.3, 0.8],
    y: [0, 0.03, 0.05, 0, -0.04, -0.02],
    surface: { x: sx, y: sy, cp: TOY_CP },
    reference: { name: 'FVM ref', points: sx.map((x, i) => [x, sy[i], TOY_CP[i] + 0.1]) },
    metrics: { rms: 0.1, max: 0.1, n: 6, cl: 0.3, cd: 0.01, cl_ref: 0.35, collapsed: false },
    timing: { train_s: 12.5, infer_ms: 1.5 },
    log: { round: [1, 2], time_s: [5, 12], loss: [1e-2, 1e-3], loss_f: [1e-2, 1e-3], loss_b: [1e-3, 1e-4], cl: [0.2, 0.3] },
    snapshots: [
      { round: 1, cp: [0, 0, 0, 0, 0, 0] },
      { round: 2, cp: TOY_CP },
    ],
    field: { width: 2, height: 1, xmin: -0.4, xmax: 1.4, ymin: -0.5, ymax: 0.5, cp: [0, 1], u: [1, 1], v: [0, 0] },
    ...extra,
  };
}

describe('caseToResult', () => {
  it('gives a solve-response-shaped object the flow view can use', () => {
    const c = toy();
    const r = caseToResult(c);
    expect(r.solver).toBe('pinn');
    expect(r.code).toBe('Toy');
    expect(r.alpha).toBe(4);
    expect(r.mach).toBe(0.4);
    expect(r.x).toBe(c.x);
    expect(r.y).toBe(c.y);
    expect(r.cl).toBe(0.3);
    expect(r.solve_ms).toBe(1.5);
    expect(r.surface.cx).toBe(c.surface.x);
    expect(r.surface.cy).toBe(c.surface.y);
    expect(r.surface.cp).toBe(c.surface.cp);
  });
  it('marks the faces before the minimum-x face as the upper surface', () => {
    expect(caseToResult(toy()).surface.upper).toEqual([true, true, true, false, false, false]);
  });
  it('keeps the field by reference (no copy of about 1 MB of numbers)', () => {
    const c = toy();
    expect(caseToResult(c).field).toBe(c.field);
  });
  it('copes with a case that has no field or no timing', () => {
    const r = caseToResult(toy({ field: undefined, timing: { train_s: null, infer_ms: null } }));
    expect(r.field).toBeNull();
    expect(r.solve_ms).toBeNull();
  });
});

describe('replay', () => {
  it('snapshotResult swaps in the Cp of that round and nothing else', () => {
    const c = toy();
    const r = snapshotResult(c, 0);
    expect(r.surface.cp).toBe(c.snapshots[0].cp);
    expect(r.surface.cx).toBe(c.surface.x);
    expect(r.field).toBe(c.field);
  });
  it('replayCurve returns the round number, its Cp and the Cl logged for that round', () => {
    const c = toy();
    const p = replayCurve(c, 0);
    expect(p.round).toBe(1);
    expect(p.cp).toBe(c.snapshots[0].cp);
    expect(p.cl).toBe(0.2);
    expect(replayCurve(c, 1).cl).toBe(0.3);
  });
  it('replayCurve clamps the index and returns null for a case without snapshots', () => {
    expect(replayCurve(toy(), 99).round).toBe(2);
    expect(replayCurve(toy(), -3).round).toBe(1);
    expect(replayCurve(toy({ snapshots: [] }), 0)).toBeNull();
    expect(replayCurve(toy({ snapshots: undefined }), 0)).toBeNull();
  });
  it('replayCurve gives a null Cl when the log has no entry for that round', () => {
    expect(replayCurve(toy({ log: null }), 0).cl).toBeNull();
  });
});

describe('cpError', () => {
  it('is the offset for a shifted curve and zero for the reference itself (points outside 0.02 <= x <= 0.98 are left out)', () => {
    const c = toy();
    const shifted = cpError(c.surface, c.reference.points, TOY_CP);
    expect(shifted.n).toBe(5); // the face at x = 0.0 is left out
    expect(shifted.rms).toBeCloseTo(0.1, 12);
    expect(shifted.max).toBeCloseTo(0.1, 12);
    const exact = cpError(c.surface, c.reference.points, c.reference.points.map((p) => p[2]));
    expect(exact.rms).toBeCloseTo(0, 12);
  });
  it('returns null numbers when no reference point is usable', () => {
    expect(cpError(toy().surface, [], TOY_CP)).toEqual({ rms: null, max: null, n: 0 });
  });
});

describe('roundRms', () => {
  it('computes the Cp error of every snapshot against the reference, and nulls without a reference', () => {
    const c = toy();
    const r = roundRms(c);
    expect(r).toHaveLength(2);
    expect(r[1]).toBeCloseTo(0.1, 12);
    expect(r[0]).toBeGreaterThan(0.1);
    expect(roundRms(toy({ reference: null }))).toEqual([null, null]);
  });
});

// The case as a public deployment serves it (backend aeronet.pinn_cases.strip_fvm): no FVM points anywhere, the numbers kept.
function publicToy(extra = {}) {
  const { reference, ...rest } = toy();
  return { ...rest, fvm_ref_name: reference.name, rms_by_round: [0.25, 0.1], fvm_same_errors: { name: 'FVM same', rms: 0.04, max: 0.2, cl: 0.4 }, fvm_curves_withheld: true, ...extra };
}

describe('a case served without the FVM curves (public deployment)', () => {
  it('has the per-round Cp RMS from the server instead of computing it from the points', () => {
    expect(roundRms(publicToy())).toEqual([0.25, 0.1]);
    expect(roundRms(publicToy({ rms_by_round: undefined }))).toEqual([null, null]);
    expect(roundRms(publicToy({ rms_by_round: [0.1] }))).toEqual([null, null]); // wrong length: not used
    expect(roundRms(publicToy({ rms_by_round: [0.25, null] }))).toEqual([0.25, null]);
  });
  it('is described by fvmView with names and numbers but no points', () => {
    const v = fvmView(publicToy());
    expect(v.withheld).toBe(true);
    expect(v.ref).toEqual({ name: 'FVM ref', points: null });
    expect(v.same).toEqual({ name: 'FVM same', rms: 0.04, max: 0.2, cl: 0.4, points: null });
  });
  it('fvmView of a full case keeps the points, and of a case with no reference says so', () => {
    const full = fvmView(toy({ fvm_same_mesh: { name: 'FVM s', rms: 0.1, max: 0.2, cl: 0.4, points: [[0.1, 0.02, 0.3]] } }));
    expect(full.withheld).toBe(false);
    expect(full.ref.points).toHaveLength(6);
    expect(full.same.points).toEqual([[0.1, 0.02, 0.3]]);
    expect(fvmView(toy({ reference: null, fvm_same_mesh: null }))).toEqual({ withheld: false, ref: null, same: null });
  });
  it('still gives the flow view and the wall-Cp chart range (no crash, the PINN and panel curves set the range)', () => {
    const c = publicToy({ panel: { cl: 0.5, surface: { x: [0.5], y: [0], cp: [-2.2] } } });
    expect(caseToResult(c).surface.cp).toEqual(TOY_CP);
    const [lo, hi] = yRange({ c, cp: TOY_CP, show: { ref: true, same: true, panel: true } });
    expect(lo).toBeLessThanOrEqual(-2.2 - 0.1);
    expect(hi).toBeGreaterThanOrEqual(1.5);
    expect(() => yRange({ c: publicToy(), cp: TOY_CP, show: { ref: true, same: true, panel: false } })).not.toThrow();
  });
});

describe('rowsFor', () => {
  const rows = [
    { experiment: 'E1', method: 'pinn', source: 'measured' },
    { experiment: 'E4', method: 'panel', source: 'measured' },
    { experiment: 'E4', method: 'fvm', source: 'quoted' },
  ];
  it('filters by experiment and keeps the export order', () => {
    expect(rowsFor(rows, 'E4').map((r) => r.method)).toEqual(['panel', 'fvm']);
    expect(rowsFor(rows, 'E9')).toEqual([]);
  });
  it('returns every row for "all", nothing for a missing table', () => {
    expect(rowsFor(rows, 'all')).toHaveLength(3);
    expect(rowsFor(rows)).toHaveLength(3);
    expect(rowsFor(null, 'E4')).toEqual([]);
  });
  it('applies a source filter on top', () => {
    expect(rowsFor(rows, 'E4', 'quoted').map((r) => r.method)).toEqual(['fvm']);
    expect(rowsFor(rows, 'all', 'measured')).toHaveLength(2);
  });
});

describe('findRow', () => {
  const rows = [
    { experiment: 'E4', method: 'pinn', mesh: '50x25', infer_ms: 1.2 },
    { experiment: 'E4', method: 'pinn', mesh: '100x50', infer_ms: 3.7 },
    { experiment: 'E1', method: 'pinn', mesh: '50x25' },
  ];
  it('finds a row by its identifying fields, or returns null', () => {
    expect(findRow(rows, { experiment: 'E4', method: 'pinn', mesh: '100x50' }).infer_ms).toBe(3.7);
    expect(findRow(rows, { experiment: 'E4', method: 'fvm' })).toBeNull();
    expect(findRow(null, { experiment: 'E4' })).toBeNull();
  });
});

describe('labels and grouping', () => {
  it('shortVariant keeps what follows the shape, mesh, angle and Mach number', () => {
    expect(shortVariant('NACA 0012 · 100×50 · 4° · Ma 0.4 · grid-seq., positivity on')).toBe('grid-seq., positivity on');
    expect(shortVariant('NACA 0012 · 100×50 · 4° · Ma 0.4')).toBe('');
  });
  it('groups cases by shape (a space in the NACA code) and orders them by mesh size, then angle', () => {
    const list = [
      { id: 'c', shape: 'NACA2412', mesh: '100x50', alpha: 4, label: 'b' },
      { id: 'a', shape: 'NACA0012', mesh: '100x50', alpha: 4, label: 'a' },
      { id: 'b', shape: 'NACA0012', mesh: '50x25', alpha: 4, label: 'z' },
      { id: 'd', shape: 'NACA2412', mesh: '100x50', alpha: -4, label: 'c' },
    ];
    const g = caseGroups(list);
    expect(g.map((x) => x.heading)).toEqual(['NACA 0012', 'NACA 2412']);
    expect(g[0].cases.map((x) => x.id)).toEqual(['b', 'a']); // 50x25 (1250 cells) before 100x50
    expect(g[1].cases.map((x) => x.id)).toEqual(['d', 'c']); // -4 deg before 4 deg
  });
  it('methodLabel names the known methods and passes others through', () => {
    expect(methodLabel('pinn_gs_nopos')).toBe('PINN grid-seq., positivity off');
    expect(methodLabel('thin_airfoil')).toBe('Thin-airfoil theory');
    expect(methodLabel('something_new')).toBe('something_new');
  });
});
