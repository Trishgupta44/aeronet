// Pure helpers for the PINN Lab: turn an exported case (app/backend/data/pinn_cases/<id>.json, see roadmap 5.7) into the objects the
// flow view and the charts use, replay its training rounds, and select rows of the results table. No React, no DOM.

const clampIndex = (list, k) => Math.max(0, Math.min(list.length - 1, Math.round(k)));

/**
 * The case as a solve-response-shaped object (what FlowCanvas reads: field, x, y, alpha, cl, surface).
 * `upper` is true for the faces before the minimum-x face: the faces run counter-clockwise from the trailing edge,
 * so the leading edge splits them into upper and lower. The field is kept by reference (about 1 MB of numbers per case).
 */
export function caseToResult(c) {
  const s = c.surface;
  let iMin = 0;
  s.x.forEach((x, i) => {
    if (x < s.x[iMin]) iMin = i;
  });
  return {
    solver: 'pinn',
    code: c.shape,
    alpha: c.alpha,
    mach: c.mach,
    x: c.x,
    y: c.y,
    surface: { cx: s.x, cy: s.y, cp: s.cp, upper: s.x.map((_, i) => i < iMin) },
    cl: c.metrics?.cl ?? null,
    cd: c.metrics?.cd ?? null,
    solve_ms: c.timing?.infer_ms ?? null,
    field: c.field ?? null,
  };
}

/** The same object with the surface Cp of training round `k` (an index into `snapshots`). */
export function snapshotResult(c, k) {
  const r = caseToResult(c);
  const snap = c.snapshots?.[k];
  return snap ? { ...r, surface: { ...r.surface, cp: snap.cp } } : r;
}

/** Round `k` of the replay: `{round, cp, cl}` (Cl from the training log, null when the log has no entry), or null without snapshots. */
export function replayCurve(c, k) {
  const snaps = c.snapshots;
  if (!snaps || !snaps.length) return null;
  const snap = snaps[clampIndex(snaps, k)];
  const at = c.log?.round ? c.log.round.indexOf(snap.round) : -1;
  return { round: snap.round, cp: snap.cp, cl: at >= 0 ? c.log.cl[at] ?? null : null };
}

/**
 * The Cp error of a wall curve against reference points `[x, y, cp]`: for every reference point with 0.02 <= x <= 0.98 the nearest
 * predicted face (in x, y) is used. Same definition as research/metrics.py::cp_error, so the final round reproduces the exported RMS.
 * `surface` is `{x, y}` (case form) or `{cx, cy}` (result form).
 */
export function cpError(surface, refPoints, cp) {
  const xs = surface.x ?? surface.cx;
  const ys = surface.y ?? surface.cy;
  let sum = 0;
  let max = 0;
  let n = 0;
  for (const [rx, ry, rcp] of refPoints) {
    if (rx < 0.02 || rx > 0.98) continue;
    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = (xs[i] - rx) ** 2 + (ys[i] - ry) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    const e = cp[best] - rcp;
    sum += e * e;
    max = Math.max(max, Math.abs(e));
    n += 1;
  }
  return n ? { rms: Math.sqrt(sum / n), max, n } : { rms: null, max: null, n: 0 };
}

/**
 * Cp RMS against the reference for every snapshot (null each when there is nothing to compare with). Computed here from the reference
 * points when the case has them; a public deployment serves the case WITHOUT the FVM points (NNfoil has no licence for redistribution)
 * but with `rms_by_round`, the same numbers computed on the server before the points were dropped.
 */
export function roundRms(c) {
  const snaps = c.snapshots ?? [];
  if (c.reference?.points?.length) return snaps.map((s) => cpError(c.surface, c.reference.points, s.cp).rms);
  if (Array.isArray(c.rms_by_round) && c.rms_by_round.length === snaps.length) return c.rms_by_round.map((v) => (Number.isFinite(v) ? v : null));
  return snaps.map(() => null);
}

/**
 * What a case says about the finite-volume (FVM) data, in either form: the full case (`reference` and `fvm_same_mesh` with their
 * points) or the public one (`fvm_ref_name`, `fvm_same_errors` and `fvm_curves_withheld`, no points). `points` is null whenever the curve
 * is not there, so a chart can simply skip it. `ref` and `same` are null when the case has no such reference at all.
 */
export function fvmView(c) {
  const refPoints = c.reference?.points?.length ? c.reference.points : null;
  const refName = c.reference?.name ?? c.fvm_ref_name ?? null;
  const s = c.fvm_same_mesh ?? c.fvm_same_errors ?? null;
  return {
    withheld: !!c.fvm_curves_withheld,
    ref: refName ? { name: refName, points: refPoints } : null,
    same: s ? { name: s.name, rms: s.rms, max: s.max, cl: s.cl, points: s.points?.length ? s.points : null } : null,
  };
}

/** Rows of the results table for one experiment ('all' or empty: every row), optionally one source (measured, quoted, fitted). */
export function rowsFor(results, experiment, source) {
  if (!results) return [];
  return results.filter((r) => (!experiment || experiment === 'all' || r.experiment === experiment) && (!source || source === 'all' || r.source === source));
}

/** The first row that matches every given field (e.g. `{experiment: 'E4', method: 'pinn', mesh: '100x50'}`), or null. */
export function findRow(results, match) {
  if (!results) return null;
  return results.find((r) => Object.entries(match).every(([k, v]) => r[k] === v)) ?? null;
}

/** What a case label says after "shape · mesh · angle · Mach": the variant (grid-seq., positivity, seed...). Empty when there is none. */
export function shortVariant(label) {
  return (label ?? '').split(' · ').slice(4).join(' · ');
}

const cells = (mesh) => {
  const m = /^(\d+)x(\d+)$/.exec(mesh ?? '');
  return m ? Number(m[1]) * Number(m[2]) : 0;
};

/** "NACA0012" -> "NACA 0012"; other names unchanged. */
export const shapeHeading = (shape) => (shape ?? '').replace(/^NACA(\d)/, 'NACA $1');

/** The case list grouped by shape (NACA codes first, then the rest), each group ordered by mesh size, then angle, then label. */
export function caseGroups(list) {
  const by = new Map();
  for (const c of list ?? []) {
    if (!by.has(c.shape)) by.set(c.shape, []);
    by.get(c.shape).push(c);
  }
  const rank = (s) => (String(s).startsWith('NACA') ? 0 : 1);
  return [...by.entries()]
    .sort(([a], [b]) => rank(a) - rank(b) || String(a).localeCompare(String(b)))
    .map(([shape, cases]) => ({
      shape,
      heading: shapeHeading(shape),
      cases: [...cases].sort((a, b) => cells(a.mesh) - cells(b.mesh) || a.alpha - b.alpha || String(a.label).localeCompare(String(b.label))),
    }));
}

const METHODS = {
  pinn: 'PINN, positivity on',
  pinn_nopos: 'PINN, positivity off',
  pinn_gs: 'PINN grid-seq., positivity on',
  pinn_gs_nopos: 'PINN grid-seq., positivity off',
  fvm: 'FVM',
  panel: 'Panel method',
  thin_airfoil: 'Thin-airfoil theory',
  neuralfoil: 'NeuralFoil (viscous)',
  pinn_stoprule: 'PINN, stop-rule replay',
  pinn_nopos_stoprule: 'PINN (positivity off), stop-rule replay',
  pinn_stoprule_replay: 'PINN, stop-rule replay',
  pinn_stoprule_online: 'PINN, stop-rule online run',
};
export const methodLabel = (m) => METHODS[m] ?? m;

export const EXPERIMENTS = {
  B0: 'Baselines: FVM on coarser meshes, panel method and thin-airfoil theory against the FVM 800×400 reference.',
  E1: 'Positivity on or off, three seeds, NACA 0012 on the 50×25 mesh: neither wins, and most runs drift late.',
  E2: 'Mesh study: PINN against FVM on the same mesh (50×25, 100×50, 200×100), grid sequencing included.',
  E2b: 'How far each PINN is from the FVM on its own mesh: it is not imitating the coarse solver.',
  E3: 'NACA 2412 at Mach 0.3 (quoted from the paper) and four angles; one run collapsed.',
  E4: 'Speed: one forward pass of a trained network against a panel solve and quoted FVM times. Not a like-for-like comparison.',
  E5: 'The cylinder at Mach 0.4: PINN against panel method and FVM meshes.',
  E6: 'The stop rule: a reference-free rule that failed to stop drifted runs, replayed on 15 held-out runs.',
};
