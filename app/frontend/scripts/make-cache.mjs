// Regenerate the saved answers used by the offline fallback (safe demo mode).
//
//   (backend running, with results/ available)   node scripts/make-cache.mjs [http://localhost:8000]
//
// Writes public/cache/bundle.json: { generated_utc, api_version, entries: { <cacheKey>: <answer> } }. The requests are built
// with the same cacheKey() the browser uses, for every airfoil preset and every shape in the library at its starting angle
// and Mach number, plus the PINN showdown, the trade-off chart, the viscous check and the Cl polars, the results table and
// the E1 round-by-round data of the Results page, a handful of PINN Lab cases (the list offered offline is cut down to exactly those cases). Nothing is invented: every entry is a
// real answer of the running backend. The only change made to a saved PINN case is a smaller flow field (2x2 block means,
// noted in the case as `field_note`), to keep the file size reasonable.

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cacheKey } from '../src/offline/cacheKey.js';

const API = (process.argv[2] ?? 'http://localhost:8000').replace(/\/$/, '');
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'cache', 'bundle.json');
const POLAR_ALPHAS = Array.from({ length: 23 }, (_, k) => k - 10); // same as hooks/usePolar.js
const RE_CHOICES = [1e5, 5e5, 1e6, 3e6, 1e7]; // same as ViscousCard
const GRID = 100; // field resolution of the saved solutions (keeps the file small)
// PINN Lab cases kept for offline use: the headline grid-sequenced case, the 50x25 pair (drift, and the run that stayed good),
// the collapsed NACA 2412 run and its working retry.
const PINN_CASES = [
  'naca0012_100x50_a4_m04_gs',
  'naca0012_50x25_a4_m04',
  'naca0012_50x25_a4_m04_nopos',
  'naca2412_100x50_a4_m03',
  'naca2412_100x50_a4_m03_nopos',
];

const entries = {};
const skipped = [];
const kept = [];
// Answers the running backend cannot give (the viscous check needs NeuralFoil, an optional install) are carried over from the
// bundle written earlier, when it has them: those are real answers of an earlier run, never invented ones.
let previous = {};
try {
  previous = JSON.parse(await readFile(OUT, 'utf8')).entries ?? {};
} catch {
  /* no earlier bundle */
}

async function call(path, body) {
  const res = await fetch(API + path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${(await res.text()).slice(0, 120)}`);
  return res.json();
}

async function save(path, body) {
  const key = cacheKey(path, body);
  try {
    entries[key] = await call(path, body);
  } catch (e) {
    if (previous[key] !== undefined) {
      entries[key] = previous[key];
      kept.push(key);
    } else {
      skipped.push(e.message);
    }
  }
}

// numbers: 6 decimals everywhere, 4 in the pressure-field arrays (they dominate the file size)
const round = (v, d) => (v !== 0 && Math.abs(v) < 0.01 ? Number(v.toPrecision(6)) : Math.round(v * 10 ** d) / 10 ** d); // tiny values (training losses) keep 6 significant digits
function compact(x, d = 6) {
  if (typeof x === 'number') return Number.isInteger(x) ? x : round(x, d);
  if (Array.isArray(x)) return x.map((v) => compact(v, d));
  if (x && typeof x === 'object') return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, compact(v, k === 'field' ? 4 : d)]));
  return x;
}

/** The flow field at half resolution: every 2x2 block becomes one cell (mean of its free cells, null if all four are in the body). */
function halveField(f) {
  const w2 = Math.floor(f.width / 2);
  const h2 = Math.floor(f.height / 2);
  const dx = (f.xmax - f.xmin) / f.width;
  const dy = (f.ymax - f.ymin) / f.height;
  const out = { width: w2, height: h2, xmin: f.xmin, xmax: f.xmin + 2 * w2 * dx, ymax: f.ymax, ymin: f.ymax - 2 * h2 * dy, cp: [], u: [], v: [] };
  for (const key of ['cp', 'u', 'v']) {
    for (let j = 0; j < h2; j++) {
      for (let i = 0; i < w2; i++) {
        const vals = [];
        for (const [a, b] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
          const v = f[key][(2 * j + b) * f.width + 2 * i + a];
          if (v !== null && v !== undefined) vals.push(v);
        }
        out[key].push(vals.length ? Math.round((vals.reduce((p, q) => p + q, 0) / vals.length) * 1000) / 1000 : null); // 3 decimals
      }
    }
  }
  return out;
}

const health = await call('/api/health');
for (const p of ['/api/solvers', '/api/airfoils', '/api/shapes', '/api/reference/sample', '/api/showdown', '/api/tradeoff']) await save(p);

const airfoils = entries[cacheKey('/api/airfoils')];
const shapes = entries[cacheKey('/api/shapes')];
const sample = entries[cacheKey('/api/reference/sample')];

// PINN Lab: the case list (only the saved cases), the saved cases, and the results table
await save('/api/results');
await save('/api/results/e1_rounds'); // the Results page's training-trajectory chart
const pinnList = await call('/api/pinn/cases');
const pinnSaved = [];
for (const id of PINN_CASES) {
  if (!pinnList.some((c) => c.id === id)) {
    skipped.push(`PINN case ${id} is not in /api/pinn/cases`);
    continue;
  }
  try {
    const c = await call(`/api/pinn/cases/${id}`);
    const full = c.field ? c.field.width : 0;
    if (c.field) {
      c.field = halveField(c.field);
      c.field_note = `Offline copy: the flow field is saved at half resolution (${c.field.width}x${c.field.height} cells, 2x2 block means of the ${full}-wide field).`;
    }
    entries[cacheKey(`/api/pinn/cases/${id}`)] = c;
    pinnSaved.push(id);
  } catch (e) {
    skipped.push(e.message);
  }
}
entries[cacheKey('/api/pinn/cases')] = pinnList.filter((c) => pinnSaved.includes(c.id));

for (const a of airfoils) {
  const base = { solver: 'panel', code: a.code, alpha: 4, mach: 0.4, grid_width: GRID, reference: null };
  await save('/api/solve', base);
  if (sample && sample.condition?.code === a.code) {
    await save('/api/solve', { ...base, alpha: sample.condition.alpha, mach: sample.condition.mach, reference: sample.points });
  }
  await save('/api/polar', { code: a.code, alphas: POLAR_ALPHAS, mach: 0.4 });
  for (const re of RE_CHOICES) await save('/api/viscous', { code: a.code, alpha: 4, re });
}
for (const s of shapes) {
  const ground = s.ground ?? null;
  await save('/api/solve_shape', { points: s.points, alpha: 4, mach: 0, lifting: s.lifting, ground, grid_width: GRID, reference: null });
  await save('/api/polar', { points: s.points, alphas: POLAR_ALPHAS, mach: 0, lifting: s.lifting, ground });
}

const bundle = { generated_utc: new Date().toISOString(), api_version: health.version, source: API, entries: compact(entries) };
await mkdir(dirname(OUT), { recursive: true });
const text = JSON.stringify(bundle);
await writeFile(OUT, text);
console.log(`${Object.keys(entries).length} saved answers, ${(text.length / 1e6).toFixed(2)} MB -> ${OUT}`);
if (kept.length) console.log(`kept ${kept.length} answers from the earlier bundle (the backend could not answer them now): ${kept.join(', ')}`);
if (skipped.length) console.log(`skipped ${skipped.length}:\n  ${skipped.join('\n  ')}`);
