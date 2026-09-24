// Run: node prototype/test_solver.js
const S = require('./solver.js');
let fail = 0;
function check(name, ok, info) { console.log((ok ? 'PASS ' : 'FAIL ') + name + '  ' + info); if (!ok) fail++; }

// 1. symmetric airfoil at 0 deg must give zero lift
let r = S.solve('0012', 0, 0);
check('NACA0012 alpha=0 -> Cl ~ 0', Math.abs(r.Cl) < 1e-3, 'Cl=' + r.Cl.toFixed(5));

// 2. Cl slope: thin-airfoil theory says 2*pi*alpha*(1+~0.77 t) for a thickened airfoil
r = S.solve('0012', 4, 0);
check('NACA0012 alpha=4 -> Cl in 0.44..0.50', r.Cl > 0.44 && r.Cl < 0.50, 'Cl=' + r.Cl.toFixed(4) + ' thin=' + r.thin.toFixed(4));

// 3. d'Alembert: drag of inviscid flow should be ~0
check('NACA0012 alpha=4 -> |Cd| < 0.01 (d\'Alembert)', Math.abs(r.Cd) < 0.01, 'Cd=' + r.Cd.toFixed(5));

// 4. cambered airfoil has lift at alpha=0 (NACA2412 alpha_L0 ~ -2.1 deg -> Cl ~ 0.23-0.26)
r = S.solve('2412', 0, 0);
check('NACA2412 alpha=0 -> Cl in 0.20..0.30', r.Cl > 0.20 && r.Cl < 0.30, 'Cl=' + r.Cl.toFixed(4) + ' thin=' + r.thin.toFixed(4));

// 5. antisymmetry: Cl(-a) = -Cl(a) for symmetric airfoil
const a = S.solve('0012', 6, 0), b = S.solve('0012', -6, 0);
check('antisymmetry', Math.abs(a.Cl + b.Cl) < 1e-6, 'Cl(6)=' + a.Cl.toFixed(4) + ' Cl(-6)=' + b.Cl.toFixed(4));

// 6. Prandtl-Glauert at Ma 0.4 raises Cl by 1/sqrt(1-0.16)
const c = S.solve('0012', 4, 0.4), d = S.solve('0012', 4, 0);
check('Prandtl-Glauert ratio', Math.abs(c.Cl / d.Cl - 1 / Math.sqrt(0.84)) < 1e-6, 'ratio=' + (c.Cl / d.Cl).toFixed(4));

// 7. Kutta condition: trailing-edge Cp of the two surfaces agree
r = S.solve('0012', 4, 0);
check('Kutta: Cp at TE panels agree', Math.abs(r.Cp[0] - r.Cp[r.N - 1]) < 0.05, 'Cp_up=' + r.Cp[0].toFixed(3) + ' Cp_lo=' + r.Cp[r.N - 1].toFixed(3));

// 8. stagnation point: max Cp near 1 at the leading edge
check('max Cp near 1 (stagnation)', Math.max(...r.Cp) > 0.95 && Math.max(...r.Cp) < 1.02, 'maxCp=' + Math.max(...r.Cp).toFixed(3));

// 9. speed
const t = Date.now(); for (let i = 0; i < 10; i++) S.solve('2412', 3, 0.3);
check('solve is interactive (<150 ms each)', (Date.now() - t) / 10 < 150, ((Date.now() - t) / 10).toFixed(1) + ' ms per solve');

console.log(fail ? '\n' + fail + ' FAILED' : '\nall passed');
process.exit(fail ? 1 : 0);
