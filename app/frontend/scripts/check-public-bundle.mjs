// Check the saved results (public/cache/bundle.json) of a static demo or a container build against the mode the build is meant for.
//
//   node scripts/check-public-bundle.mjs                      include mode (the default): the NNfoil reference data MUST be in the bundle
//   node scripts/check-public-bundle.mjs --withhold           strict mode: the bundle must contain NO FVM data
//   AERONET_WITHHOLD_FVM_DATA=1 node scripts/check-public-bundle.mjs     same as --withhold
//   node scripts/check-public-bundle.mjs [--withhold] path/to/bundle.json
//
// Why two modes. NNfoil's author gave permission to show the finite-volume (FVM) reference data in this project, so a public build
// includes it by default (with credit on the pages). The strict mode stays for a deployment that must not carry it (for example if that
// permission is ever withdrawn): there the bundle has to be made by a backend that withholds the data (deploy/prepare-public.ps1 -Withhold).
// Either way the check refuses a bundle that does not match the chosen mode, so a build cannot silently be the other kind.
// Exit code 1 means: regenerate the bundle (see docs/DEPLOY.md).

import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const withhold = args.includes('--withhold') || ['1', 'true', 'yes', 'on'].includes((process.env.AERONET_WITHHOLD_FVM_DATA ?? '').trim().toLowerCase());
const fileArg = args.find((a) => !a.startsWith('--'));
const file = resolve(fileArg ?? resolve(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'cache', 'bundle.json'));
let bundle;
try {
  bundle = JSON.parse(await readFile(file, 'utf8'));
} catch (e) {
  console.error(`Cannot read ${file}: ${e.message}`);
  process.exit(1);
}

/** Path of the first non-empty `points` list anywhere in `node` (the format of an FVM curve in an exported case), or null. */
function findPoints(node, path = '') {
  if (Array.isArray(node)) {
    for (let i = 0; i < node.length; i++) {
      if (node[i] && typeof node[i] === 'object') {
        const hit = findPoints(node[i], `${path}[${i}]`);
        if (hit) return hit;
      }
    }
    return null;
  }
  for (const [k, v] of Object.entries(node ?? {})) {
    if (k === 'points' && Array.isArray(v) && v.length) return `${path}.points`;
    if (v && typeof v === 'object') {
      const hit = findPoints(v, `${path}.${k}`);
      if (hit) return hit;
    }
  }
  return null;
}

const e = bundle.entries ?? {};
const caseEntries = Object.entries(e).filter(([k, c]) => k.startsWith('/api/pinn/cases/') && c && typeof c === 'object');
const sd = e['/api/showdown'];
const caps = e['/api/capabilities'];
const problems = [];

if (withhold) {
  // ---- strict mode: no FVM data of any kind --------------------------------------------------------------------------------------
  if (e['/api/reference/sample'] !== undefined) problems.push('contains the bundled FVM sample (/api/reference/sample)');
  if (Object.keys(e).some((k) => k.startsWith('/api/solve|') && /\|ref[1-9]\d*$/.test(k))) problems.push('contains a solution computed with the FVM sample as reference overlay');
  if (!sd) problems.push('has no /api/showdown entry');
  else {
    if (sd.reference?.curve) problems.push('showdown: the FVM reference curve is present');
    if ((sd.meshes ?? []).some((m) => m.fvm?.curve)) problems.push('showdown: an FVM mesh curve is present');
    if (sd.fvm_curves_withheld !== true) problems.push('showdown: not marked fvm_curves_withheld');
  }
  // The exported PINN cases (PINN Lab): the `reference` points (FVM 800x400) and the `fvm_same_mesh` points are FVM Cp curves, and no case
  // may carry any `points` list at all (the FVM curve format). The numbers computed against them are fine.
  for (const [key, c] of caseEntries) {
    const name = key.slice('/api/pinn/cases/'.length);
    if (c.reference?.points?.length) problems.push(`PINN case ${name}: the FVM reference points are present`);
    if (c.fvm_same_mesh?.points?.length) problems.push(`PINN case ${name}: the same-mesh FVM points are present`);
    const stray = findPoints(c);
    if (stray) problems.push(`PINN case ${name}: a "points" curve is present at ${stray}`);
  }
  if (caps?.fvm_curves !== false || caps?.public_build !== true) problems.push('capabilities: not marked as a withholding build (fvm_curves false, public_build true)');
} else {
  // ---- include mode: the reference data is meant to be there; a stripped bundle would silently show less than the live app ------------
  if (e['/api/reference/sample'] === undefined) problems.push('has no bundled FVM sample (/api/reference/sample): it was made by a backend that withholds the data');
  if (!sd) problems.push('has no /api/showdown entry');
  else {
    if (!sd.reference?.curve) problems.push('showdown: the FVM reference curve is missing');
    if (sd.fvm_curves_withheld === true) problems.push('showdown: marked fvm_curves_withheld');
  }
  if (!caseEntries.some(([, c]) => c.reference?.points?.length)) problems.push('no saved PINN case carries the FVM reference points');
  if (caseEntries.some(([, c]) => c.fvm_curves_withheld === true)) problems.push('a saved PINN case is marked fvm_curves_withheld');
  if (caps && (caps.fvm_curves === false || caps.public_build === true)) problems.push('capabilities: marked as a withholding build');
}

if (problems.length) {
  const other = withhold ? 'a strict (--withhold)' : 'an include-mode';
  console.error(`${file} does not fit ${other} build:\n  - ${problems.join('\n  - ')}\nRegenerate it: deploy/prepare-public.ps1${withhold ? ' -Withhold' : ''} (docs/DEPLOY.md).`);
  process.exit(1);
}
console.log(withhold
  ? `ok (strict): ${file} has no FVM data (${Object.keys(e).length} entries, generated ${bundle.generated_utc})`
  : `ok (include): ${file} carries the reference data as intended (${Object.keys(e).length} entries, generated ${bundle.generated_utc})`);
