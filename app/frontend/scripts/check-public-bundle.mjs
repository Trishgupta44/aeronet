// Refuse to build a public site from saved results that contain NNfoil's finite-volume (FVM) Cp data.
//
//   node scripts/check-public-bundle.mjs [path/to/bundle.json]
//
// A public build uses a bundle made by scripts/make-cache.mjs against a backend started with AERONET_WITHHOLD_FVM_DATA=1
// (deploy/prepare-public.ps1 does exactly that). The bundle in the repository may be a private one with the FVM sample and
// curves in it; this check stops such a bundle from being published by accident. Exit code 1 means: regenerate it.

import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const file = resolve(process.argv[2] ?? resolve(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'cache', 'bundle.json'));
let bundle;
try {
  bundle = JSON.parse(await readFile(file, 'utf8'));
} catch (e) {
  console.error(`Cannot read ${file}: ${e.message}`);
  process.exit(1);
}

const problems = [];
const e = bundle.entries ?? {};
if (e['/api/reference/sample'] !== undefined) problems.push('contains the bundled FVM sample (/api/reference/sample)');
if (Object.keys(e).some((k) => k.startsWith('/api/solve|') && /\|ref[1-9]\d*$/.test(k))) problems.push('contains a solution computed with the FVM sample as reference overlay');
const sd = e['/api/showdown'];
if (!sd) problems.push('has no /api/showdown entry');
else {
  if (sd.reference?.curve) problems.push('showdown: the FVM reference curve is present');
  if ((sd.meshes ?? []).some((m) => m.fvm?.curve)) problems.push('showdown: an FVM mesh curve is present');
  if (sd.fvm_curves_withheld !== true) problems.push('showdown: not marked fvm_curves_withheld');
}
if (e['/api/capabilities']?.public_build !== true) problems.push('capabilities: public_build is not true');

if (problems.length) {
  console.error(`${file} is not safe for a public build:\n  - ${problems.join('\n  - ')}\nRegenerate it with deploy/prepare-public.ps1.`);
  process.exit(1);
}
console.log(`ok: ${file} has no FVM data (${Object.keys(e).length} entries, generated ${bundle.generated_utc})`);
