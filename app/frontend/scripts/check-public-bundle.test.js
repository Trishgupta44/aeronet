import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const script = new URL('./check-public-bundle.mjs', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');

const safe = () => ({
  generated_utc: '2026-11-14T00:00:00Z',
  entries: {
    '/api/capabilities': { public_build: true, fvm_sample: false, fvm_curves: false },
    '/api/showdown': { fvm_curves_withheld: true, reference: { curve: null }, meshes: [{ mesh: '100x50', fvm: { rms: 0.046, curve: null }, pinn: [] }] },
    '/api/solve|panel|0012|4|0.4|ref0': { cl: 0.5 },
  },
});

function run(bundle) {
  const file = join(mkdtempSync(join(tmpdir(), 'bundle-')), 'bundle.json');
  writeFileSync(file, JSON.stringify(bundle));
  try {
    return { code: 0, out: execFileSync('node', [script, file], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) };
  } catch (e) {
    return { code: e.status, out: `${e.stderr}${e.stdout}` };
  }
}

describe('check-public-bundle', () => {
  it('accepts a bundle made by a backend that withholds the FVM data', () => {
    expect(run(safe())).toMatchObject({ code: 0 });
  });
  it('rejects the bundled FVM sample', () => {
    const b = safe();
    b.entries['/api/reference/sample'] = { points: [[0, 0, 1]] };
    const r = run(b);
    expect(r.code).toBe(1);
    expect(r.out).toContain('FVM sample');
  });
  it('rejects an FVM curve in the showdown', () => {
    const b = safe();
    b.entries['/api/showdown'].meshes[0].fvm.curve = { x: [0], cp: [0], upper: [true] };
    expect(run(b).code).toBe(1);
  });
  it('rejects a solution computed with the FVM sample as overlay', () => {
    const b = safe();
    b.entries['/api/solve|panel|0012|4|0.4|ref800'] = { comparison: { rms: 0.01 } };
    expect(run(b).code).toBe(1);
  });
  it('rejects a bundle that was not made in public mode', () => {
    const b = safe();
    b.entries['/api/capabilities'].public_build = false;
    expect(run(b).code).toBe(1);
  });
});
