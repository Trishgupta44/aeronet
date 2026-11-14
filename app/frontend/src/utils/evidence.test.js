import { describe, expect, it } from 'vitest';
import {
  LOG_SPEED_MAX_MS, LOG_SPEED_MIN_MS, cylinderBars, driftedRuns, failureItems, headlineTiles, logPos, meshStudy, meshSummary, oneThreadMs, speedBars, trajectories, trajectorySummary,
} from './evidence.js';

// Rows shaped like /api/results (results/summary.csv), only the columns these helpers read.
const row = (experiment, method, mesh, o = {}) => ({ experiment, case: 'NACA0012_a4', method, mesh, rms_cp: null, cl: null, cl_ref: null, train_s: null, infer_ms: null, source: 'measured', notes: '', ...o });
const E4NOTE = (a, b) => `infer_ms = full-field forward pass ... 8 threads, mean over 100 calls after warm-up: ${a} +/- 0.05 ms (n=100); 1 thread: ${b} +/- 0.05 ms (n=100); wall-only evaluate_wall (Cp, Cl, Cd): 8 threads 0.844 +/- 0.09 ms (n=100); 1 thread 0.888 +/- 0.027 ms (n=100)`;
const ROWS = [
  row('E2', 'fvm', '50x25', { rms_cp: 0.108087, cl: 0.426457, cl_ref: 0.519329 }),
  row('E2', 'fvm', '100x50', { rms_cp: 0.045985, cl: 0.480267, cl_ref: 0.519329 }),
  row('E2', 'fvm', '200x100', { rms_cp: 0.018648 }),
  row('E2', 'fvm', '400x200', { rms_cp: 0.005927 }),
  row('E2', 'panel', '140 panels', { rms_cp: 0.013503 }),
  row('E2', 'pinn', '50x25', { rms_cp: 0.073405 }),
  row('E2', 'pinn_nopos', '50x25', { rms_cp: 0.032526 }),
  row('E2', 'pinn', '100x50', { rms_cp: 0.014451, cl: 0.516286, cl_ref: 0.519329 }),
  row('E2', 'pinn_gs', '100x50', { rms_cp: 0.016216 }),
  row('E2', 'pinn_gs_nopos', '200x100', { rms_cp: 0.009213 }),
  row('E4', 'pinn', '50x25', { infer_ms: 1.2651, notes: E4NOTE('1.265', '3.570') }),
  row('E4', 'pinn', '100x50', { infer_ms: 3.7535, notes: E4NOTE('3.754', '13.418') }),
  row('E4', 'pinn_gs_nopos', '200x100', { infer_ms: 16.7665, notes: E4NOTE('16.766', '59.310') }),
  row('E4', 'panel', '140 panels', { infer_ms: 3.6416, notes: 'infer_ms = panel.solve(...)' }),
  row('E4', 'neuralfoil', 'large', { infer_ms: 1.1618 }),
  row('E4', 'fvm', '50x25', { infer_ms: 9000, source: 'quoted' }),
  row('E4', 'fvm', '400x200', { infer_ms: 285000, source: 'quoted' }),
  row('E5', 'pinn', '50x25', { case: 'Cylinder_a0', rms_cp: 0.140321 }),
  row('E5', 'pinn_nopos', '50x25', { case: 'Cylinder_a0', rms_cp: 0.13367 }),
  row('E5', 'panel', '140 panels', { case: 'Cylinder_a0', rms_cp: 0.1877 }),
  row('E5', 'panel', '140 panels (Ma 0)', { case: 'Cylinder_a0', rms_cp: 0.307245 }),
  row('E5', 'fvm', '200x100', { case: 'Cylinder_a0', rms_cp: 0.078989 }),
  row('E3', 'pinn', '100x50', { case: 'NACA2412_a4', rms_cp: 15.639509, cl: -0, cl_ref: 0.760497 }),
  row('E3', 'pinn', '100x50', { case: 'NACA2412_a0', rms_cp: 0.0157, cl: 0.2468, cl_ref: 0.2651 }),
];

describe('headlineTiles', () => {
  const t = headlineTiles(ROWS);
  it('makes four tiles, each with a source tag', () => {
    expect(t.map((x) => x.id)).toEqual(['cp100', 'cp200', 'cl100', 'speed100']);
    expect(t.every((x) => x.tag === 'measured' && x.ok)).toBe(true);
  });
  it('computes the Cp ratios from the rows (3.2x at 100x50, 2.0x at 200x100)', () => {
    expect(t[0].value).toBe('3.2×');
    expect(t[0].compare).toContain('0.0145');
    expect(t[0].compare).toContain('0.0460');
    expect(t[1].value).toBe('2.0×');
    expect(t[1].compare).toContain('0.0092');
    expect(t[1].compare).toContain('0.0186');
  });
  it('gives the lift error against the reference, and the same-mesh FVM error beside it', () => {
    expect(t[2].value).toBe('0.6%');
    expect(t[2].compare).toContain('0.516');
    expect(t[2].compare).toContain('0.519');
    expect(t[2].compare).toContain('7.5%');
  });
  it('quotes the inference time next to the panel solve, and says one condition only', () => {
    expect(t[3].value).toBe('3.75 ms');
    expect(t[3].compare).toContain('3.64 ms');
    expect(t[3].note).toMatch(/one condition/i);
  });
  it('shows n/a, never a guess, when a row is missing', () => {
    const none = headlineTiles(ROWS.filter((r) => !(r.experiment === 'E2' && r.method === 'fvm' && r.mesh === '100x50')));
    expect(none[0].value).toBe('n/a');
    expect(none[0].ok).toBe(false);
    expect(none[2].compare).not.toContain('7.5%'); // the FVM Cl row is gone too
    expect(headlineTiles(null).every((x) => x.value === 'n/a' && !x.ok)).toBe(true);
  });
});

describe('meshStudy', () => {
  const m = meshStudy(ROWS);
  it('has one group per FVM mesh with its PINN runs next to it', () => {
    expect(m.groups.map((g) => g.mesh)).toEqual(['50x25', '100x50', '200x100', '400x200']);
    expect(m.groups[0].fvm).toBeCloseTo(0.108087);
    expect(m.groups[0].pinn.map((p) => p.rms)).toEqual([0.073405, 0.032526]);
    expect(m.groups[1].pinn.map((p) => p.method)).toEqual(['pinn', 'pinn_gs']);
    expect(m.groups[3].pinn).toEqual([]); // no PINN at 400x200
  });
  it('keeps the panel level and flags grid sequencing and positivity for the markers', () => {
    expect(m.panel).toBeCloseTo(0.013503);
    const [on, off] = m.groups[0].pinn;
    expect(on.positivity).toBe(true);
    expect(off.positivity).toBe(false);
    expect(m.groups[1].pinn[1].gridSeq).toBe(true);
  });
});

describe('trajectories', () => {
  const run = (seed, positivity, ys) => ({ run_id: `r${seed}${positivity}`, method: positivity ? 'pinn' : 'pinn_nopos', positivity, seed, mesh: '50x25', rounds: ys.map((_, i) => i + 1), cp_rms: ys, cl: ys.map(() => 0.5) });
  const e1 = { runs: [run(0, true, [0.19, 0.037, 0.03, 0.028, 0.03, 0.07]), run(0, false, [0.2, 0.04, 0.027, 0.026, 0.027, 0.033])] };
  const tr = trajectories(e1, ROWS);
  it('gives one series per run with the shared round axis', () => {
    expect(tr.series).toHaveLength(2);
    expect(tr.xs).toEqual([1, 2, 3, 4, 5, 6]);
    expect(tr.series[0].label).toBe('positivity on, seed 0');
    expect(tr.series[0].ys[5]).toBe(0.07);
  });
  it('takes the FVM 50x25 error as the reference level and marks the stop window at rounds 3 to 5', () => {
    expect(tr.fvm50).toBeCloseTo(0.108087);
    expect(tr.band).toEqual({ from: 3, to: 5 });
  });
  it('counts drifted runs as final RMS above 2x the best round', () => {
    expect(driftedRuns(e1)).toEqual({ drifted: 1, total: 2 });
  });
  it('is empty without data', () => {
    expect(trajectories(null, ROWS).series).toEqual([]);
    expect(driftedRuns(null)).toEqual({ drifted: 0, total: 0 });
  });
});

describe('cylinderBars', () => {
  it('orders the four bars incompressible, panel+PG, PINN on, PINN off', () => {
    const b = cylinderBars(ROWS);
    expect(b.map((x) => x.id)).toEqual(['circle', 'panel_pg', 'pinn', 'pinn_nopos']);
    expect(b.map((x) => x.value)).toEqual([0.307245, 0.1877, 0.140321, 0.13367]);
    expect(b.map((x) => x.kind)).toEqual(['model', 'model', 'pinn', 'pinn']);
  });
  it('drops a bar whose row is missing', () => {
    expect(cylinderBars(ROWS.filter((r) => r.mesh !== '140 panels (Ma 0)')).map((x) => x.id)).not.toContain('circle');
  });
});

describe('speed', () => {
  it('reads the 1 thread time out of the E4 note, not the wall-only one', () => {
    expect(oneThreadMs(ROWS[10])).toBeCloseTo(3.57);
    expect(oneThreadMs(ROWS[13])).toBeNull();
  });
  it('builds a bar per timing with the right label kind', () => {
    const bars = speedBars(ROWS);
    const by = Object.fromEntries(bars.map((b) => [b.id, b]));
    expect(by['pinn-100x50-8'].ms).toBeCloseTo(3.7535);
    expect(by['pinn-100x50-1'].ms).toBeCloseTo(13.418);
    expect(by.panel.kind).toBe('measured');
    expect(by.neuralfoil.kind).toBe('viscous');
    expect(by['fvm-400x200'].kind).toBe('quoted');
    expect(by['fvm-400x200'].ms).toBe(285000);
    expect(bars.every((b) => b.ms > 0)).toBe(true);
  });
  it('maps milliseconds to a 0..1 position on the log axis', () => {
    expect(logPos(LOG_SPEED_MIN_MS)).toBe(0);
    expect(logPos(LOG_SPEED_MAX_MS)).toBe(1);
    expect(logPos(10)).toBeCloseTo(2 / 7);
    expect(logPos(1e-9)).toBe(0);
  });
});

describe('failureItems', () => {
  it('lists the four things that did not work, each with its source', () => {
    const f = failureItems(ROWS, { runs: [{ cp_rms: [0.03, 0.2] }, { cp_rms: [0.03, 0.031] }] });
    expect(f.map((x) => x.id)).toEqual(['stoprule', 'coldstart', 'drift', 'naca2412']);
    expect(f[0].value).toBe('0 of 8');
    expect(f[2].value).toBe('1 of 2'); // computed from the rounds file
    expect(f[3].value).toContain('NACA 2412'); // from the E3 row with Cl about 0
    expect(f.every((x) => x.source)).toBe(true);
  });
  it('leaves out the computed claim when the data is missing', () => {
    const f = failureItems([], null);
    expect(f.find((x) => x.id === 'naca2412').value).toBe('1 run');
    expect(f.find((x) => x.id === 'drift').value).toBe('most runs');
  });
});

describe('summaries', () => {
  it('says whether the PINN is lower than the FVM on every mesh that has one', () => {
    const s = meshSummary(meshStudy(ROWS));
    expect(s.allLower).toBe(true);
    expect(s.lines.map((l) => l.mesh)).toEqual(['50x25', '100x50', '200x100']);
    expect(s.lines[0]).toMatchObject({ pinnLo: 0.032526, pinnHi: 0.073405 });
    const worse = meshStudy(ROWS.map((r) => (r.experiment === 'E2' && r.method === 'pinn' && r.mesh === '50x25' ? { ...r, rms_cp: 0.2 } : r)));
    expect(meshSummary(worse).allLower).toBe(false);
  });
  it('summarises the trajectories: spread at round 3, drifted runs, worst final value', () => {
    const run = (ys) => ({ rounds: ys.map((_, i) => i + 1), cp_rms: ys });
    const s = trajectorySummary({ runs: [run([0.2, 0.04, 0.03, 0.03, 0.2]), run([0.2, 0.04, 0.027, 0.03, 0.03])] });
    expect(s).toEqual({ r3: [0.027, 0.03], drifted: 1, total: 2, worstFinal: 0.2 });
    expect(trajectorySummary(null)).toBeNull();
  });
});
