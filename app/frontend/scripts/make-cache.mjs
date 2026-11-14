// Regenerate the saved answers used by the offline fallback (safe demo mode).
//
//   (backend running, with results/ available)   node scripts/make-cache.mjs [http://localhost:8000]
//
// Writes public/cache/bundle.json: { generated_utc, api_version, entries: { <cacheKey>: <answer> } }. The requests are built
// with the same cacheKey() the browser uses, for every airfoil preset and every shape in the library at its starting angle
// and Mach number, plus the PINN showdown, the trade-off chart, the viscous check and the Cl polars. Nothing is invented:
// every entry is a real answer of the running backend.

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cacheKey } from '../src/offline/cacheKey.js';

const API = (process.argv[2] ?? 'http://localhost:8000').replace(/\/$/, '');
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'cache', 'bundle.json');
const POLAR_ALPHAS = Array.from({ length: 23 }, (_, k) => k - 10); // same as hooks/usePolar.js
const RE_CHOICES = [1e5, 5e5, 1e6, 3e6, 1e7]; // same as ViscousCard
const GRID = 100; // field resolution of the saved solutions (keeps the file small)

const entries = {};
const skipped = [];

async function call(path, body) {
  const res = await fetch(API + path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${(await res.text()).slice(0, 120)}`);
  return res.json();
}

async function save(path, body) {
  try {
    entries[cacheKey(path, body)] = await call(path, body);
  } catch (e) {
    skipped.push(e.message);
  }
}

// numbers: 6 decimals everywhere, 4 in the pressure-field arrays (they dominate the file size)
const round = (v, d) => Math.round(v * 10 ** d) / 10 ** d;
function compact(x, d = 6) {
  if (typeof x === 'number') return Number.isInteger(x) ? x : round(x, d);
  if (Array.isArray(x)) return x.map((v) => compact(v, d));
  if (x && typeof x === 'object') return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, compact(v, k === 'field' ? 4 : d)]));
  return x;
}

const health = await call('/api/health');
for (const p of ['/api/solvers', '/api/airfoils', '/api/shapes', '/api/capabilities', '/api/reference/sample', '/api/showdown', '/api/tradeoff']) await save(p);

const airfoils = entries[cacheKey('/api/airfoils')];
const shapes = entries[cacheKey('/api/shapes')];
const sample = entries[cacheKey('/api/reference/sample')];

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
if (skipped.length) console.log(`skipped ${skipped.length}:\n  ${skipped.join('\n  ')}`);
