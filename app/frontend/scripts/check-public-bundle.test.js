import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const script = new URL('./check-public-bundle.mjs', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
const CASE = '/api/pinn/cases/naca0012_100x50_a4_m04_gs';
const PTS = [[0.1, 0.05, -0.3], [0.5, 0.06, -0.8]];

// A bundle made by a backend that WITHHOLDS the FVM data (strict mode).
const strictBundle = () => ({
  generated_utc: '2026-11-14T00:00:00Z',
  entries: {
    '/api/capabilities': { public_build: true, fvm_sample: false, fvm_curves: false },
    '/api/showdown': { fvm_curves_withheld: true, reference: { curve: null }, meshes: [{ mesh: '100x50', fvm: { rms: 0.046, curve: null }, pinn: [] }] },
    '/api/solve|panel|0012|4|0.4|ref0': { cl: 0.5 },
    '/api/pinn/cases': [{ id: 'naca0012_100x50_a4_m04_gs', rms: 0.016 }],
    [CASE]: {
      id: 'naca0012_100x50_a4_m04_gs',
      surface: { x: [0.5], y: [0.05], cp: [-0.4] },
      fvm_ref_name: 'FVM 800×400',
      fvm_same_errors: { name: 'FVM 100×50', rms: 0.046, max: 0.26, cl: 0.48 },
      rms_by_round: [0.2, 0.016],
      fvm_curves_withheld: true,
    },
  },
});

// A bundle made by a normal backend (include mode: the reference data is part of it).
const includeBundle = () => ({
  generated_utc: '2026-11-14T00:00:00Z',
  entries: {
    '/api/capabilities': { public_build: false, fvm_sample: true, fvm_curves: true },
    '/api/reference/sample': { name: 'NACA0012 200×100 FVM sample', points: PTS },
    '/api/showdown': { fvm_curves_withheld: false, reference: { curve: { x: [0.1], cp: [-0.3], upper: [true] } }, meshes: [{ mesh: '100x50', fvm: { rms: 0.046, curve: { x: [0.1], cp: [-0.3], upper: [true] } }, pinn: [] }] },
    '/api/solve|panel|0012|4|0.4|ref200': { comparison: { rms: 0.01 } },
    [CASE]: { id: 'naca0012_100x50_a4_m04_gs', surface: { x: [0.5], y: [0.05], cp: [-0.4] }, reference: { name: 'FVM 800×400', points: PTS }, fvm_same_mesh: { name: 'FVM 100×50', points: PTS, rms: 0.046 } },
  },
});

function run(bundle, ...flags) {
  const file = join(mkdtempSync(join(tmpdir(), 'bundle-')), 'bundle.json');
  writeFileSync(file, JSON.stringify(bundle));
  try {
    return { code: 0, out: execFileSync('node', [script, ...flags, file], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) };
  } catch (e) {
    return { code: e.status, out: `${e.stderr}${e.stdout}` };
  }
}
const strict = (b) => run(b, '--withhold');

describe('check-public-bundle, strict mode (--withhold): no FVM data allowed', () => {
  it('accepts a bundle made by a backend that withholds the FVM data', () => {
    expect(strict(strictBundle())).toMatchObject({ code: 0 });
  });
  it('is also switched on by AERONET_WITHHOLD_FVM_DATA=1', () => {
    const file = join(mkdtempSync(join(tmpdir(), 'bundle-')), 'bundle.json');
    writeFileSync(file, JSON.stringify(includeBundle()));
    let code = 0;
    try {
      execFileSync('node', [script, file], { env: { ...process.env, AERONET_WITHHOLD_FVM_DATA: '1' }, stdio: 'ignore' });
    } catch (e) {
      code = e.status;
    }
    expect(code).toBe(1);
  });
  it('rejects the bundled FVM sample', () => {
    const b = strictBundle();
    b.entries['/api/reference/sample'] = { points: [[0, 0, 1]] };
    const r = strict(b);
    expect(r.code).toBe(1);
    expect(r.out).toContain('FVM sample');
  });
  it('rejects an FVM curve in the showdown', () => {
    const b = strictBundle();
    b.entries['/api/showdown'].meshes[0].fvm.curve = { x: [0], cp: [0], upper: [true] };
    expect(strict(b).code).toBe(1);
  });
  it('rejects a solution computed with the FVM sample as overlay', () => {
    const b = strictBundle();
    b.entries['/api/solve|panel|0012|4|0.4|ref800'] = { comparison: { rms: 0.01 } };
    expect(strict(b).code).toBe(1);
  });
  it('rejects the FVM 800x400 reference points in a saved PINN case (the exported-case format)', () => {
    const b = strictBundle();
    b.entries[CASE].reference = { name: 'FVM 800×400', points: PTS };
    const r = strict(b);
    expect(r.code).toBe(1);
    expect(r.out).toContain('reference points');
  });
  it('rejects the same-mesh FVM points in a saved PINN case', () => {
    const b = strictBundle();
    b.entries[CASE].fvm_same_mesh = { name: 'FVM 100×50', rms: 0.046, points: PTS };
    const r = strict(b);
    expect(r.code).toBe(1);
    expect(r.out).toContain('same-mesh');
  });
  it('rejects a stray "points" curve anywhere in a saved case, but accepts the numbers computed against the FVM', () => {
    const b = strictBundle();
    b.entries[CASE].extra = { nested: [{ points: PTS }] };
    expect(strict(b).code).toBe(1);
    expect(strict(strictBundle()).code).toBe(0); // fvm_same_errors, rms_by_round and fvm_ref_name are numbers and a label: allowed
  });
  it('accepts shape outlines named "points" outside the cases (the library shapes use that key)', () => {
    const b = strictBundle();
    b.entries['/api/shapes'] = [{ name: 'circle', points: [[1, 0], [0, 1]] }];
    expect(strict(b).code).toBe(0);
  });
  it('rejects a bundle that was not made in withholding mode', () => {
    const b = strictBundle();
    b.entries['/api/capabilities'].public_build = false;
    expect(strict(b).code).toBe(1);
    expect(strict(includeBundle()).code).toBe(1);
  });
});

describe('check-public-bundle, include mode (the default): the reference data must be there', () => {
  it('accepts a bundle made by a normal backend', () => {
    const r = run(includeBundle());
    expect(r.code).toBe(0);
    expect(r.out).toContain('include');
  });
  it('rejects a bundle without the FVM sample', () => {
    const b = includeBundle();
    delete b.entries['/api/reference/sample'];
    const r = run(b);
    expect(r.code).toBe(1);
    expect(r.out).toContain('withholds');
  });
  it('rejects a bundle whose showdown has no FVM reference curve', () => {
    const b = includeBundle();
    b.entries['/api/showdown'].reference.curve = null;
    expect(run(b).code).toBe(1);
  });
  it('rejects a bundle whose saved PINN cases carry no reference points', () => {
    const b = includeBundle();
    delete b.entries[CASE].reference;
    expect(run(b).code).toBe(1);
  });
  it('rejects a withheld bundle (made in strict mode) in include mode', () => {
    expect(run(strictBundle()).code).toBe(1);
  });
  it('rejects a bundle marked as withholding even if the data happens to be there', () => {
    const b = includeBundle();
    b.entries['/api/capabilities'].public_build = true;
    expect(run(b).code).toBe(1);
  });
});
