import { useRef } from 'react';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';

function drawCp(g, w, h, result, reference, pinned) {
  if (!result) return;
  const m = { l: 46, r: 14, t: 12, b: 34 };
  const s = result.surface;
  const cps = s.cp.slice();
  if (reference) reference.points.forEach((p) => cps.push(p[2]));
  if (pinned) pinned.surface.cp.forEach((c) => cps.push(c));
  const lo = Math.min(-1, Math.floor((Math.min(...cps) - 0.1) * 2) / 2);
  const hi = 1.5;
  const X = (x) => m.l + x * (w - m.l - m.r);
  const Y = (v) => m.t + ((v - lo) / (hi - lo)) * (h - m.t - m.b); // inverted axis: suction is up

  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  g.font = '12px system-ui';
  g.fillStyle = cssVar('--muted');
  g.strokeStyle = cssVar('--grid');
  g.lineWidth = 1;
  for (let v = Math.ceil(lo); v <= hi; v += 0.5) {
    g.beginPath();
    g.moveTo(m.l, Y(v));
    g.lineTo(w - m.r, Y(v));
    g.stroke();
    g.textAlign = 'right';
    g.fillText(v.toFixed(1), m.l - 6, Y(v) + 4);
  }
  for (let x = 0; x <= 1.001; x += 0.25) {
    g.beginPath();
    g.moveTo(X(x), m.t);
    g.lineTo(X(x), h - m.b);
    g.stroke();
    g.textAlign = 'center';
    g.fillText(x.toFixed(2), X(x), h - m.b + 16);
  }
  g.fillText('x / chord', (m.l + w - m.r) / 2, h - 6);
  g.save();
  g.translate(12, (m.t + h - m.b) / 2);
  g.rotate(-Math.PI / 2);
  g.textAlign = 'center';
  g.fillText('Cp (suction is up)', 0, 0);
  g.restore();

  if (pinned) {
    // the pinned result as a grey dashed ghost, behind the live curves
    const side = (flag) => pinned.surface.cx.map((x, i) => [x, pinned.surface.cp[i], pinned.surface.upper[i]]).filter((q) => q[2] === flag).sort((a, b) => a[0] - b[0]);
    g.save();
    g.setLineDash([6, 4]);
    g.strokeStyle = cssVar('--muted');
    g.lineWidth = 1.6;
    for (const arr of [side(true), side(false)]) {
      g.beginPath();
      arr.forEach((q, i) => (i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]))));
      g.stroke();
    }
    g.restore();
  }

  const up = [];
  const lw = [];
  s.cx.forEach((x, i) => (s.upper[i] ? up : lw).push([x, s.cp[i]]));
  [[up, '--up'], [lw, '--lo']].forEach(([arr, col]) => {
    arr.sort((a, b) => a[0] - b[0]);
    g.strokeStyle = cssVar(col);
    g.lineWidth = 2;
    g.beginPath();
    arr.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
    g.stroke();
  });
  if (reference) {
    g.strokeStyle = cssVar('--ref');
    g.lineWidth = 1.2;
    reference.points.forEach((p) => {
      g.beginPath();
      g.arc(X(p[0]), Y(p[2]), 2.6, 0, 6.2832);
      g.stroke();
    });
  }
}

export default function CpCard({ result, reference, pinned = null, idp = '' }) {
  const ref = useRef(null);
  useCanvasDraw(ref, (w) => Math.max(260, Math.min(360, w * 0.45)), (g, w, h) => drawCp(g, w, h, result, reference, pinned), [result, reference, pinned]);
  const c = reference && result?.comparison;
  return (
    <section className="card">
      <h2>Surface pressure distribution</h2>
      <div>
        <canvas id={`${idp}cp`} ref={ref} aria-label="Cp versus chord position" role="img" />
      </div>
      <div className="legend">
        <span><i style={{ background: 'var(--up)' }} />upper surface</span>
        <span><i style={{ background: 'var(--lo)' }} />lower surface</span>
        {reference && <span><i style={{ background: 'transparent', border: '1.5px solid var(--ref)' }} />reference (FVM)</span>}
        {pinned && <span><i style={{ background: 'transparent', border: '1.5px dashed var(--muted)' }} />pinned: {pinned.label}</span>}
      </div>
      <div className="status" id={`${idp}cmpStatus`} role="status">
        {c && c.rms !== null && (
          <>
            Difference vs {reference.name}: <b>RMS {c.rms.toFixed(3)}</b>, max {c.max.toFixed(3)} ({c.n} points). Only meaningful when airfoil,
            angle and Mach match the reference (bundled NACA0012 sample: 0012, 4°, Mach 0.4).
          </>
        )}
      </div>
    </section>
  );
}
