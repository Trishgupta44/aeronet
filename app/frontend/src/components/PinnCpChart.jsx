import { useRef } from 'react';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';

const M = { l: 46, r: 14, t: 12, b: 34 };

// Split a wall curve at its leading edge (the minimum-x point): the faces before it are the upper side, as in caseToResult.
function sides(xs, cps) {
  let iMin = 0;
  xs.forEach((x, i) => {
    if (x < xs[iMin]) iMin = i;
  });
  const up = [];
  const lo = [];
  xs.forEach((x, i) => (i < iMin ? up : lo).push([x, cps[i]]));
  return [up, lo].map((a) => a.sort((p, q) => p[0] - q[0]));
}

function strokeSides(g, X, Y, curveSides, colors, width, dash) {
  g.save();
  g.setLineDash(dash || []);
  g.lineWidth = width;
  curveSides.forEach((pts, k) => {
    g.strokeStyle = colors[k];
    g.beginPath();
    pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
    g.stroke();
  });
  g.restore();
}

/** The y range shown: from the reference, the panel curve and the same-mesh FVM; the PINN only widens it when it did not collapse. */
export function yRange({ c, cp, show }) {
  const vals = [];
  if (show.ref && c.reference) c.reference.points.forEach((p) => vals.push(p[2]));
  if (show.panel && c.panel?.surface) vals.push(...c.panel.surface.cp);
  if (show.same && c.fvm_same_mesh) c.fvm_same_mesh.points.forEach((p) => vals.push(p[2]));
  if (!vals.length || !c.metrics?.collapsed) vals.push(...cp);
  const lo = Math.min(-1, Math.floor((Math.min(...vals) - 0.1) * 2) / 2);
  const hi = Math.max(1.5, Math.ceil(Math.max(...vals) * 2) / 2);
  return [lo, hi];
}

function draw(g, w, h, { c, cp, show }) {
  const [lo, hi] = yRange({ c, cp, show });
  const X = (x) => M.l + x * (w - M.l - M.r);
  const Y = (v) => M.t + ((v - lo) / (hi - lo)) * (h - M.t - M.b); // inverted axis: suction is up
  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  g.font = '12px system-ui';
  g.fillStyle = cssVar('--muted');
  g.strokeStyle = cssVar('--grid');
  g.lineWidth = 1;
  g.textAlign = 'right';
  for (let v = Math.ceil(lo); v <= hi; v += lo < -2.5 ? 1 : 0.5) {
    g.beginPath();
    g.moveTo(M.l, Y(v));
    g.lineTo(w - M.r, Y(v));
    g.stroke();
    g.fillText(v.toFixed(1), M.l - 6, Y(v) + 4);
  }
  g.textAlign = 'center';
  for (let x = 0; x <= 1.001; x += 0.25) {
    g.beginPath();
    g.moveTo(X(x), M.t);
    g.lineTo(X(x), h - M.b);
    g.stroke();
    g.fillText(x.toFixed(2), X(x), h - M.b + 16);
  }
  g.fillText('x / chord', (M.l + w - M.r) / 2, h - 6);
  g.save();
  g.translate(12, (M.t + h - M.b) / 2);
  g.rotate(-Math.PI / 2);
  g.fillText('Cp (suction is up)', 0, 0);
  g.restore();

  g.save();
  g.beginPath();
  g.rect(M.l, M.t, w - M.l - M.r, h - M.t - M.b); // anything outside the plot (a collapsed PINN at Cp -16) is clipped, and said so below
  g.clip();

  if (show.panel && c.panel?.surface) {
    const p = c.panel.surface;
    strokeSides(g, X, Y, sides(p.x, p.cp), [cssVar('--muted'), cssVar('--muted')], 1.5, [6, 4]);
  }
  if (show.ref && c.reference) {
    g.strokeStyle = cssVar('--ref');
    g.lineWidth = 1;
    c.reference.points.forEach((p) => {
      g.beginPath();
      g.arc(X(p[0]), Y(p[2]), w < 500 ? 1.2 : 1.8, 0, 6.2832);
      g.stroke();
    });
  }
  if (show.same && c.fvm_same_mesh) {
    g.fillStyle = cssVar('--ok');
    c.fvm_same_mesh.points.forEach((p) => g.fillRect(X(p[0]) - 2.5, Y(p[2]) - 2.5, 5, 5));
  }
  const s = c.surface;
  strokeSides(g, X, Y, sides(s.x, cp), [cssVar('--up'), cssVar('--lo')], 2.2);
  g.restore();

  const out = cp.filter((v) => v < lo || v > hi);
  if (out.length) {
    g.font = '600 12px system-ui';
    g.textAlign = 'left';
    g.fillStyle = cssVar('--flow-stagnation');
    const a = Math.min(...cp);
    const b = Math.max(...cp);
    g.fillText(`PINN Cp ${a === b ? a.toFixed(2) : `${a.toFixed(2)} to ${b.toFixed(2)}`}: off this chart`, M.l + 8, M.t + 16);
  }
}

/** The wall Cp of one training round (`cp`) against the references the case brings; `show` is {ref, same, panel}. */
export default function PinnCpChart({ c, cp, show }) {
  const ref = useRef(null);
  useCanvasDraw(ref, (w) => Math.max(260, Math.min(380, w * 0.62)), (g, w, h) => draw(g, w, h, { c, cp, show }), [c, cp, show.ref, show.same, show.panel]);
  return <canvas id="pinncp" ref={ref} role="img" aria-label="Wall pressure coefficient of the PINN against the FVM reference and the panel method" />;
}
