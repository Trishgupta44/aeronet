// Pure helpers for the Results page: every number on it is computed here from the rows of results/summary.csv (GET /api/results),
// the E1 round data (GET /api/results/e1_rounds) or the E4 notes. A missing row gives "n/a" (never a guess). No React, no DOM.
import { fmt, fmtMs, meshLabel } from './format.js';
import { findRow } from './pinnCase.js';

const NA = 'n/a';
const PINN_METHODS = ['pinn', 'pinn_nopos', 'pinn_gs', 'pinn_gs_nopos'];
const cells = (mesh) => {
  const m = /^(\d+)x(\d+)$/.exec(mesh ?? '');
  return m ? Number(m[1]) * Number(m[2]) : 0;
};
const e2 = (rows, method, mesh) => findRow(rows, { experiment: 'E2', method, mesh });
const e4 = (rows, method, mesh) => findRow(rows, { experiment: 'E4', method, mesh });
const has = (...vals) => vals.every((v) => v != null && Number.isFinite(v));

// ------------------------------------------------------------------------------------------------ headline band
function ratioTile(rows, id, mesh, pinnMethod, label) {
  const fvm = e2(rows, 'fvm', mesh)?.rms_cp;
  const pinn = e2(rows, pinnMethod, mesh)?.rms_cp;
  const ok = has(fvm, pinn) && pinn > 0;
  return {
    id,
    ok,
    label,
    value: ok ? `${(fvm / pinn).toFixed(1)}×` : NA,
    compare: ok ? `Cp RMS ${fmt(pinn, 4)} (PINN) against ${fmt(fvm, 4)} (FVM), both against the FVM 800×400 solution` : 'The rows this needs are not in the results.',
    note: null,
    tag: 'measured',
    source: `summary.csv, E2, ${meshLabel(mesh)}`,
  };
}

/** The four headline tiles: `{id, ok, label, value, compare, note, tag, source}`. `rows` is /api/results (or null). */
export function headlineTiles(rows) {
  const clRow = e2(rows, 'pinn', '100x50');
  const fvmCl = e2(rows, 'fvm', '100x50');
  const clErr = (r) => (r && has(r.cl, r.cl_ref) && r.cl_ref !== 0 ? (Math.abs(r.cl - r.cl_ref) / Math.abs(r.cl_ref)) * 100 : null);
  const pinnErr = clErr(clRow);
  const fvmErr = clErr(fvmCl);
  const lift = {
    id: 'cl100',
    ok: pinnErr != null,
    label: 'lift error of the PINN on the 100×50 mesh, against the fine-mesh reference',
    value: pinnErr != null ? `${pinnErr.toFixed(1)}%` : NA,
    compare: pinnErr != null
      ? `Cl ${fmt(clRow.cl, 3)} against ${fmt(clRow.cl_ref, 3)} (FVM 800×400)${fvmErr != null ? `; the FVM on the same mesh is ${fvmErr.toFixed(1)}% off (Cl ${fmt(fvmCl.cl, 3)})` : ''}`
      : 'The rows this needs are not in the results.',
    note: null,
    tag: 'measured',
    source: 'summary.csv, E2, 100×50',
  };
  const t = e4(rows, 'pinn', '100x50');
  const panel = e4(rows, 'panel', '140 panels');
  const speedOk = has(t?.infer_ms);
  const speed = {
    id: 'speed100',
    ok: speedOk,
    label: 'to evaluate the trained PINN over the whole 100×50 field (8 threads)',
    value: speedOk ? fmtMs(t.infer_ms) : NA,
    compare: speedOk ? (has(panel?.infer_ms) ? `Panel method: ${fmtMs(panel.infer_ms)} per solve, no training` : 'No panel timing in the results.') : 'The rows this needs are not in the results.',
    note: 'A trained PINN answers one condition only; a new condition means retraining.',
    tag: 'measured',
    source: 'summary.csv, E4',
  };
  return [
    ratioTile(rows, 'cp100', '100x50', 'pinn', 'lower Cp error than the FVM on the same 100×50 mesh'),
    ratioTile(rows, 'cp200', '200x100', 'pinn_gs_nopos', 'lower Cp error than the FVM on the same 200×100 mesh'),
    lift,
    speed,
  ];
}

// ------------------------------------------------------------------------------------------------ mesh study (E2)
const VARIANT = {
  pinn: { label: 'cold start, positivity on', positivity: true, gridSeq: false },
  pinn_nopos: { label: 'cold start, positivity off', positivity: false, gridSeq: false },
  pinn_gs: { label: 'grid-sequenced, positivity on', positivity: true, gridSeq: true },
  pinn_gs_nopos: { label: 'grid-sequenced, positivity off', positivity: false, gridSeq: true },
};

/** `{groups: [{mesh, fvm, pinn: [{method, rms, label, positivity, gridSeq}]}], panel}`: the FVM bar of each mesh with the PINN runs of that mesh. */
export function meshStudy(rows) {
  const e2rows = (rows ?? []).filter((r) => r.experiment === 'E2');
  const meshes = e2rows.filter((r) => r.method === 'fvm').map((r) => r.mesh).sort((a, b) => cells(a) - cells(b));
  const groups = meshes.map((mesh) => ({
    mesh,
    fvm: e2rows.find((r) => r.method === 'fvm' && r.mesh === mesh).rms_cp,
    pinn: e2rows
      .filter((r) => PINN_METHODS.includes(r.method) && r.mesh === mesh && r.rms_cp != null)
      .map((r) => ({ method: r.method, rms: r.rms_cp, ...VARIANT[r.method] })),
  }));
  return { groups, panel: e2rows.find((r) => r.method === 'panel')?.rms_cp ?? null };
}

/** What the mesh chart says in words: `{allLower, lines: [{mesh, pinnLo, pinnHi, fvm}]}` for the meshes that have a PINN. `allLower` is true only when
 *  every PINN run of every such mesh has a lower error than the FVM there. */
export function meshSummary(study) {
  const lines = study.groups.filter((g) => g.pinn.length).map((g) => ({
    mesh: g.mesh, fvm: g.fvm, pinnLo: Math.min(...g.pinn.map((p) => p.rms)), pinnHi: Math.max(...g.pinn.map((p) => p.rms)),
  }));
  return { allLower: lines.length > 0 && lines.every((l) => l.pinnHi < l.fvm), lines };
}

// ------------------------------------------------------------------------------------------------ training trajectories (E1)
const sameRun = (r) => `positivity ${r.positivity ? 'on' : 'off'}, seed ${r.seed}`;

/** Final Cp RMS above twice the best round counts as drifted (the M3 definition). */
export function driftedRuns(e1) {
  const runs = e1?.runs ?? [];
  const drifted = runs.filter((r) => {
    const ys = r.cp_rms.filter((v) => v != null);
    return ys.length > 1 && ys[ys.length - 1] > 2 * Math.min(...ys);
  }).length;
  return { drifted, total: runs.length };
}

/** `{xs, series: [{id, label, positivity, seed, ys}], fvm50, band}` for the training-trajectory chart; the stop window is rounds 3 to 5 (M3 section 9). */
export function trajectories(e1, rows) {
  const runs = e1?.runs ?? [];
  const last = Math.max(0, ...runs.flatMap((r) => r.rounds));
  return {
    xs: Array.from({ length: last }, (_, i) => i + 1),
    series: runs.map((r) => ({ id: r.run_id, label: sameRun(r), positivity: r.positivity, seed: r.seed, ys: r.cp_rms })),
    fvm50: e2(rows, 'fvm', '50x25')?.rms_cp ?? null,
    band: { from: 3, to: 5 },
  };
}

/** `{r3: [lo, hi], drifted, total, worstFinal}`: the spread of the Cp RMS at round 3 across the runs, how many drifted and the worst final value; null without data. */
export function trajectorySummary(e1) {
  const runs = e1?.runs ?? [];
  const at3 = runs.map((r) => r.cp_rms[r.rounds.indexOf(3)]).filter((v) => v != null);
  const finals = runs.map((r) => r.cp_rms.filter((v) => v != null).pop()).filter((v) => v != null);
  if (!at3.length || !finals.length) return null;
  return { r3: [Math.min(...at3), Math.max(...at3)], ...driftedRuns(e1), worstFinal: Math.max(...finals) };
}

// ------------------------------------------------------------------------------------------------ the cylinder (E5)
const E5_BARS = [
  { id: 'circle', method: 'panel', mesh: '140 panels (Ma 0)', label: ['Incompressible', 'circle'], short: ['Circle,', 'Mach 0'], kind: 'model', detail: 'Incompressible circle (panel method at Mach 0)' },
  { id: 'panel_pg', method: 'panel', mesh: '140 panels', label: ['Panel +', 'Prandtl–Glauert'], short: ['Panel', '+ P–G'], kind: 'model', detail: 'Panel method with the Prandtl–Glauert correction, Mach 0.4' },
  { id: 'pinn', method: 'pinn', mesh: '50x25', label: ['PINN,', 'positivity on'], short: ['PINN', 'pos. on'], kind: 'pinn', detail: 'PINN, 50×25, positivity on, final round' },
  { id: 'pinn_nopos', method: 'pinn_nopos', mesh: '50x25', label: ['PINN,', 'positivity off'], short: ['PINN', 'pos. off'], kind: 'pinn', detail: 'PINN, 50×25, positivity off, final round' },
];

/** The four bars of the cylinder chart: `{id, label (two lines), kind, value, detail}`; a bar with no row is left out. */
export function cylinderBars(rows) {
  return E5_BARS.map((b) => {
    const r = findRow(rows, { experiment: 'E5', method: b.method, mesh: b.mesh });
    return r && r.rms_cp != null ? { id: b.id, label: b.label, short: b.short, kind: b.kind, value: r.rms_cp, detail: b.detail } : null;
  }).filter(Boolean);
}

// ------------------------------------------------------------------------------------------------ speed (E4)
export const LOG_SPEED_MIN_MS = 0.1;
export const LOG_SPEED_MAX_MS = 1e6;

/** Position 0..1 of a time (ms) on the log axis from 0.1 ms to 1000 s (seven decades); clamped. */
export function logPos(ms) {
  if (!(ms > 0)) return 0;
  const a = Math.log10(LOG_SPEED_MIN_MS);
  const b = Math.log10(LOG_SPEED_MAX_MS);
  return Math.min(1, Math.max(0, (Math.log10(ms) - a) / (b - a)));
}

/** The 1 thread full-field time of an E4 PINN row, read from its notes ("... 1 thread: 3.570 +/- ..."); null for rows without it. */
export function oneThreadMs(r) {
  const m = /1 thread: ([\d.]+)/.exec(r?.notes ?? '');
  return m ? Number(m[1]) : null;
}

/** The bars of the speed card: `{id, label, ms, kind: measured | viscous | quoted, note}` in reading order. */
export function speedBars(rows) {
  const bars = [];
  for (const mesh of ['50x25', '100x50', '200x100']) {
    const r = e4(rows, 'pinn', mesh) ?? (rows ?? []).find((x) => x.experiment === 'E4' && PINN_METHODS.includes(x.method) && x.mesh === mesh);
    if (!r) continue;
    if (has(r.infer_ms)) bars.push({ id: `pinn-${mesh}-8`, label: `PINN ${meshLabel(mesh)}, 8 threads`, ms: r.infer_ms, kind: 'measured', note: 'full field', trainS: r.train_s ?? null });
    const one = oneThreadMs(r);
    if (one != null) bars.push({ id: `pinn-${mesh}-1`, label: `PINN ${meshLabel(mesh)}, 1 thread`, ms: one, kind: 'measured', note: 'full field', trainS: r.train_s ?? null });
  }
  const panel = e4(rows, 'panel', '140 panels');
  if (has(panel?.infer_ms)) bars.push({ id: 'panel', label: 'Panel method', ms: panel.infer_ms, kind: 'measured', note: 'one solve, no training' });
  const nf = e4(rows, 'neuralfoil', 'large');
  if (has(nf?.infer_ms)) bars.push({ id: 'neuralfoil', label: 'NeuralFoil', ms: nf.infer_ms, kind: 'viscous', note: 'viscous, not comparable' });
  for (const mesh of ['50x25', '100x50', '200x100', '400x200']) {
    const r = e4(rows, 'fvm', mesh);
    if (has(r?.infer_ms)) bars.push({ id: `fvm-${mesh}`, label: `FVM ${meshLabel(mesh)}`, ms: r.infer_ms, kind: 'quoted', note: 'quoted from the paper, other hardware' });
  }
  return bars;
}

// ------------------------------------------------------------------------------------------------ what did not work
/**
 * The four things that did not work, calm and sourced: `{id, value, title, text, source, link: {label, href | exp}}`.
 * The drift count comes from the E1 round data and the NACA 2412 collapse from the E3 row whose Cl is about 0; the stop-rule and
 * cold-start counts are quoted from the Milestone 3 report (their data files are not part of the app) and say so in `source`.
 */
export function failureItems(rows, e1) {
  const { drifted, total } = driftedRuns(e1);
  const collapsed = (rows ?? []).find((r) => r.experiment === 'E3' && r.method === 'pinn' && r.rms_cp != null && r.rms_cp > 1);
  return [
    {
      id: 'stoprule', value: '0 of 8', title: 'The stop rule failed',
      text: 'A rule that stops training without the reference stopped none of the 8 drifted 50×25 runs. What works is a fixed early stop at rounds 3 to 5.',
      source: 'M3 report, section 9 (results/e6_stop_rule.md)', link: { label: 'See the E6 rows', exp: 'E6' },
    },
    {
      id: 'coldstart', value: '1 of 3', title: 'Cold starts can collapse',
      text: 'A cold 100×50 start without positivity collapsed in 1 of 3 seeds. Grid sequencing, starting from a 50×25 run, did not (0 of 8).',
      source: 'M3 report, section 5 (results/seed_check.csv)', link: { label: 'Open the PINN Lab', href: '#pinn' },
    },
    {
      id: 'drift', value: total ? `${drifted} of ${total}` : 'most runs', title: 'The late drift is unsolved',
      text: 'Accurate by round 3, then the 50×25 runs drift while the loss keeps falling. There is no remedy yet beyond stopping early.',
      source: total ? 'results/e1_rounds.csv, final RMS above 2x the best round' : 'M3 report, section 3', link: { label: 'See the E1 rows', exp: 'E1' },
    },
    {
      id: 'naca2412', value: collapsed ? `NACA 2412, ${fmt(collapsed.alpha_deg, 0)}°` : '1 run', title: 'One run collapsed, kept in the data',
      text: 'The first NACA 2412 run at 4° (seed 0) fell to the trivial solution (Cl about 0). Two retries worked, so it is reported, not hidden.',
      source: collapsed ? 'summary.csv, E3 (Cp RMS above 1)' : 'M3 report, section 6', link: { label: 'Open this case', href: '?pinn=naca2412_100x50_a4_m03#pinn' },
    },
  ];
}
