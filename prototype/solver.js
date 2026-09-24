/*
 * AeroNet prototype solver (STAND-IN for the PINN).
 * Hess-Smith panel method: incompressible, inviscid, 2D, plus Prandtl-Glauert for Mach.
 * Real physics, but NOT the neural network. In the final project the same functions
 * (geometry -> Cp on surface, Cp field, Cl) are supplied by the trained NNfoil model.
 *
 * Conventions: chord = 1, leading edge at (0,0), freestream speed = 1.
 * Nodes run counter-clockwise (x right, y up): trailing edge -> along upper surface -> leading edge -> lower surface -> trailing edge.
 * Outward normal = right-hand normal of the panel direction.
 */
(function (root) {
  'use strict';

  // ---------- geometry: NACA 4-digit ----------
  function naca4(code, nHalf) {
    nHalf = nHalf || 80;
    const m = parseInt(code[0], 10) / 100;
    const p = parseInt(code[1], 10) / 10;
    const t = parseInt(code.slice(2), 10) / 100;
    const camber = (x) => (m === 0 ? 0 : x < p ? (m / (p * p)) * (2 * p * x - x * x)
      : (m / ((1 - p) * (1 - p))) * ((1 - 2 * p) + 2 * p * x - x * x));
    const dcamber = (x) => (m === 0 ? 0 : x < p ? (2 * m / (p * p)) * (p - x)
      : (2 * m / ((1 - p) * (1 - p))) * (p - x));
    const thick = (x) => 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x
      + 0.2843 * x * x * x - 0.1036 * x * x * x * x); // closed trailing edge
    const pt = (k, upper) => {
      const x = 0.5 * (1 + Math.cos(Math.PI * k / nHalf));
      const yc = camber(x), phi = Math.atan(dcamber(x)), yt = thick(x);
      return upper
        ? [x - yt * Math.sin(phi), yc + yt * Math.cos(phi)]
        : [x + yt * Math.sin(phi), yc - yt * Math.cos(phi)];
    };
    const X = [], Y = [];
    for (let k = 0; k <= nHalf; k++) { const q = pt(k, true); X.push(q[0]); Y.push(q[1]); }
    for (let k = nHalf - 1; k >= 1; k--) { const q = pt(k, false); X.push(q[0]); Y.push(q[1]); }
    return { X, Y, m, p, t, camber, dcamber };
  }

  // ---------- panel influence (unit strength), local frame: x along panel, y = left (interior) ----------
  function panelGeom(X, Y) {
    const N = X.length, P = [];
    for (let j = 0; j < N; j++) {
      const j2 = (j + 1) % N;
      const dx = X[j2] - X[j], dy = Y[j2] - Y[j], l = Math.hypot(dx, dy);
      const tx = dx / l, ty = dy / l;
      P.push({ x0: X[j], y0: Y[j], l, tx, ty, nx: ty, ny: -tx, // outward normal
        cx: 0.5 * (X[j] + X[j2]), cy: 0.5 * (Y[j] + Y[j2]) });
    }
    return P;
  }

  // returns [sourceVel(x,y), vortexVel(x,y)] in global axes for unit strengths
  function influence(pj, px, py, self) {
    let us, vs;
    if (self) { us = 0; vs = -0.5; } // limit on the exterior (fluid) side
    else {
      const rx = px - pj.x0, ry = py - pj.y0;
      const xl = rx * pj.tx + ry * pj.ty;
      const yl = -rx * pj.ty + ry * pj.tx;
      us = Math.log((xl * xl + yl * yl) / ((xl - pj.l) * (xl - pj.l) + yl * yl)) / (4 * Math.PI);
      vs = (Math.atan2(yl, xl - pj.l) - Math.atan2(yl, xl)) / (2 * Math.PI);
    }
    const ux = -pj.ty, uy = pj.tx; // local y axis in global coordinates
    const sx = us * pj.tx + vs * ux, sy = us * pj.ty + vs * uy;
    const vx = -vs * pj.tx + us * ux, vy = -vs * pj.ty + us * uy;
    return [sx, sy, vx, vy];
  }

  function gauss(A, b) {
    const n = b.length;
    for (let i = 0; i < n; i++) {
      let piv = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
      [A[i], A[piv]] = [A[piv], A[i]]; [b[i], b[piv]] = [b[piv], b[i]];
      for (let r = i + 1; r < n; r++) {
        const f = A[r][i] / A[i][i];
        if (f === 0) continue;
        for (let c = i; c < n; c++) A[r][c] -= f * A[i][c];
        b[r] -= f * b[i];
      }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i];
      for (let c = i + 1; c < n; c++) s -= A[i][c] * x[c];
      x[i] = s / A[i][i];
    }
    return x;
  }

  // ---------- main solve ----------
  function solve(code, alphaDeg, mach, nHalf) {
    const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
    const geo = naca4(code, nHalf);
    const X = geo.X, Y = geo.Y, N = X.length;
    const P = panelGeom(X, Y);
    const a = alphaDeg * Math.PI / 180, Vx = Math.cos(a), Vy = Math.sin(a);

    const A = [], b = new Array(N + 1).fill(0);
    for (let i = 0; i < N; i++) A.push(new Array(N + 1).fill(0));
    A.push(new Array(N + 1).fill(0));
    const Ts = [], Tv = []; // tangential influence matrices (reused for Cp)
    for (let i = 0; i < N; i++) {
      Ts.push(new Array(N)); Tv.push(new Array(N));
      let gsum = 0;
      for (let j = 0; j < N; j++) {
        const f = influence(P[j], P[i].cx, P[i].cy, i === j);
        A[i][j] = f[0] * P[i].nx + f[1] * P[i].ny;
        gsum += f[2] * P[i].nx + f[3] * P[i].ny;
        Ts[i][j] = f[0] * P[i].tx + f[1] * P[i].ty;
        Tv[i][j] = f[2] * P[i].tx + f[3] * P[i].ty;
      }
      A[i][N] = gsum;
      b[i] = -(Vx * P[i].nx + Vy * P[i].ny);
    }
    // Kutta condition: tangential speeds at the two trailing-edge panels cancel
    let g = 0;
    for (let j = 0; j < N; j++) {
      A[N][j] = Ts[0][j] + Ts[N - 1][j];
      g += Tv[0][j] + Tv[N - 1][j];
    }
    A[N][N] = g;
    b[N] = -(Vx * (P[0].tx + P[N - 1].tx) + Vy * (P[0].ty + P[N - 1].ty));
    const sol = gauss(A, b);
    const sigma = sol.slice(0, N), gamma = sol[N];

    // surface pressure (Prandtl-Glauert scaled)
    const beta = Math.sqrt(1 - mach * mach);
    const Cp = new Array(N);
    for (let i = 0; i < N; i++) {
      let vt = Vx * P[i].tx + Vy * P[i].ty;
      for (let j = 0; j < N; j++) vt += sigma[j] * Ts[i][j] + gamma * Tv[i][j];
      Cp[i] = (1 - vt * vt) / beta;
    }
    // forces by pressure integration (chord = 1)
    let Fx = 0, Fy = 0;
    for (let i = 0; i < N; i++) { Fx -= Cp[i] * P[i].nx * P[i].l; Fy -= Cp[i] * P[i].ny * P[i].l; }
    const Cl = -Fx * Math.sin(a) + Fy * Math.cos(a);
    const Cd = Fx * Math.cos(a) + Fy * Math.sin(a); // ideal value 0 (d'Alembert)

    const ms = (typeof performance !== 'undefined' ? performance : Date).now() - t0;
    return { geo, X, Y, P, sigma, gamma, Cp, Cl, Cd, beta, a, Vx, Vy, ms, N,
      thin: thinAirfoilCl(geo, alphaDeg, mach) };
  }

  // velocity / Cp at an arbitrary point in the flow (outside the body)
  function fieldCp(res, px, py) {
    let u = res.Vx, v = res.Vy;
    for (let j = 0; j < res.N; j++) {
      const f = influence(res.P[j], px, py, false);
      u += res.sigma[j] * f[0] + res.gamma * f[2];
      v += res.sigma[j] * f[1] + res.gamma * f[3];
    }
    return (1 - (u * u + v * v)) / res.beta;
  }

  function insideBody(X, Y, px, py) { // ray casting
    let c = false;
    for (let i = 0, j = X.length - 1; i < X.length; j = i++) {
      if ((Y[i] > py) !== (Y[j] > py) && px < (X[j] - X[i]) * (py - Y[i]) / (Y[j] - Y[i]) + X[i]) c = !c;
    }
    return c;
  }

  // thin-airfoil theory: Cl = 2*pi*(alpha - alpha_L0), Prandtl-Glauert scaled
  function thinAirfoilCl(geo, alphaDeg, mach) {
    const n = 400; let s = 0;
    for (let k = 0; k < n; k++) {
      const th = (k + 0.5) * Math.PI / n, x = 0.5 * (1 - Math.cos(th));
      s += geo.dcamber(x) * (Math.cos(th) - 1) * (Math.PI / n);
    }
    const aL0 = -s / Math.PI;
    return 2 * Math.PI * (alphaDeg * Math.PI / 180 - aL0) / Math.sqrt(1 - mach * mach);
  }

  // Parse an NNfoil FVM reference file: header line, then "node x y Cp" rows.
  // x is centred on the chord (-0.5..0.5) in those files, so shift it to 0..1.
  function parseReference(text) {
    const pts = [];
    text.split(/\r?\n/).slice(1).forEach((line) => {
      const s = line.trim().split(/\s+/).map(Number);
      if (s.length >= 4 && s.slice(1, 4).every(Number.isFinite)) pts.push([s[1], s[2], s[3]]);
    });
    if (pts.length && Math.min(...pts.map((p) => p[0])) < -0.1) pts.forEach((p) => { p[0] += 0.5; });
    return pts;
  }

  // Error of the solver's surface Cp against reference points [x, y, Cp] (same airfoil, alpha, Mach assumed).
  function compareCp(res, ref) {
    const up = [], lo = [];
    res.P.forEach((p, i) => { (p.cy >= res.geo.camber(p.cx) ? up : lo).push([p.cx, res.Cp[i]]); });
    up.sort((a, b) => a[0] - b[0]); lo.sort((a, b) => a[0] - b[0]);
    const interp = (t, x) => {
      if (x < t[0][0] || x > t[t.length - 1][0]) return null;
      for (let i = 0; i + 1 < t.length; i++) {
        if (x >= t[i][0] && x <= t[i + 1][0]) {
          const f = (x - t[i][0]) / (t[i + 1][0] - t[i][0] || 1);
          return t[i][1] + f * (t[i + 1][1] - t[i][1]);
        }
      }
      return null;
    };
    let s2 = 0, mx = 0, n = 0;
    ref.forEach(([x, y, cp]) => {
      if (x < 0.02 || x > 0.98) return; // skip the very tips where the surfaces are ambiguous
      const v = interp(y >= res.geo.camber(x) ? up : lo, x);
      if (v === null) return;
      const e = v - cp; s2 += e * e; mx = Math.max(mx, Math.abs(e)); n++;
    });
    return { rms: n ? Math.sqrt(s2 / n) : NaN, max: mx, n };
  }

  const api = { naca4, solve, fieldCp, insideBody, thinAirfoilCl, parseReference, compareCp };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AeroSolver = api;
})(typeof self !== 'undefined' ? self : this);
