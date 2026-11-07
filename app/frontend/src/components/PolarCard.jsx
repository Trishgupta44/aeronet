import { useRef } from 'react';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';
import { usePolar } from '../hooks/usePolar.js';

function drawPolar(g, w, h, polar, current, extraSeries) {
  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  if (!polar) return;
  const m = { l: 46, r: 14, t: 12, b: 36 };
  const all = polar.cl.slice();
  if (polar.cl_thin) all.push(...polar.cl_thin);
  extraSeries.forEach((s) => s.points.forEach((p) => all.push(p[1])));
  if (current) all.push(current.cl);
  const lo = Math.min(-0.5, Math.floor((Math.min(...all) - 0.05) * 5) / 5);
  const hi = Math.max(0.5, Math.ceil((Math.max(...all) + 0.05) * 5) / 5);
  const a0 = polar.alpha[0];
  const a1 = polar.alpha[polar.alpha.length - 1];
  const X = (a) => m.l + ((a - a0) / (a1 - a0)) * (w - m.l - m.r);
  const Y = (c) => m.t + ((hi - c) / (hi - lo)) * (h - m.t - m.b);

  g.font = '12px system-ui';
  g.lineWidth = 1;
  g.strokeStyle = cssVar('--grid');
  g.fillStyle = cssVar('--muted');
  const step = hi - lo > 2.6 ? 0.5 : 0.25;
  for (let c = Math.ceil(lo / step) * step; c <= hi + 1e-9; c += step) {
    g.beginPath();
    g.moveTo(m.l, Y(c));
    g.lineTo(w - m.r, Y(c));
    g.stroke();
    g.textAlign = 'right';
    g.fillText(c.toFixed(2), m.l - 6, Y(c) + 4);
  }
  for (let a = Math.ceil(a0 / 4) * 4; a <= a1; a += 4) {
    g.beginPath();
    g.moveTo(X(a), m.t);
    g.lineTo(X(a), h - m.b);
    g.stroke();
    g.textAlign = 'center';
    g.fillText(String(a), X(a), h - m.b + 16);
  }
  // zero lines
  g.strokeStyle = cssVar('--muted');
  g.beginPath();
  g.moveTo(m.l, Y(0));
  g.lineTo(w - m.r, Y(0));
  g.moveTo(X(0), m.t);
  g.lineTo(X(0), h - m.b);
  g.stroke();
  g.fillStyle = cssVar('--muted');
  g.textAlign = 'center';
  g.fillText('α (deg)', (m.l + w - m.r) / 2, h - 6);
  g.save();
  g.translate(12, (m.t + h - m.b) / 2);
  g.rotate(-Math.PI / 2);
  g.fillText('Cl', 0, 0);
  g.restore();

  const line = (xs, ys, color, width, dash) => {
    g.save();
    g.setLineDash(dash);
    g.strokeStyle = color;
    g.lineWidth = width;
    g.beginPath();
    xs.forEach((a, i) => (i ? g.lineTo(X(a), Y(ys[i])) : g.moveTo(X(a), Y(ys[i]))));
    g.stroke();
    g.restore();
  };
  if (polar.cl_thin) line(polar.alpha, polar.cl_thin, cssVar('--muted'), 1.5, [6, 4]);
  line(polar.alpha, polar.cl, cssVar('--up'), 2.2, []);
  extraSeries.forEach((s) => {
    g.fillStyle = s.color ?? cssVar('--lo');
    s.points.forEach((p) => {
      g.beginPath();
      g.arc(X(p[0]), Y(p[1]), 3.2, 0, 6.2832);
      g.fill();
    });
  });
  if (current && current.alpha >= a0 && current.alpha <= a1) {
    g.fillStyle = cssVar('--heading');
    g.strokeStyle = cssVar('--plot-bg');
    g.lineWidth = 2;
    g.beginPath();
    g.arc(X(current.alpha), Y(current.cl), 5, 0, 6.2832);
    g.fill();
    g.stroke();
  }
}

/**
 * Cl against α at the current Mach number for the current airfoil code or drawn shape.
 * `source` is `{code, mach}` or `{points, lifting, ground, mach}`; `current` is `{alpha, cl}` of the live solution.
 * `extraSeries` is `[{label, points: [[alpha, cl]], color}]`: a hook for FVM and PINN points (Plan C).
 */
const NO_SERIES = [];

export default function PolarCard({ source, current, extraSeries = NO_SERIES, idp = '' }) {
  const ref = useRef(null);
  const { polar, error } = usePolar(source);
  useCanvasDraw(ref, (w) => Math.max(240, Math.min(320, w * 0.4)), (g, w, h) => drawPolar(g, w, h, polar, current, extraSeries), [polar, current?.alpha, current?.cl, extraSeries]);
  return (
    <section className="card">
      <h2>Lift curve (Cl against α)</h2>
      <canvas id={`${idp}polar`} ref={ref} role="img" aria-label="Lift coefficient against angle of attack" />
      <div className="legend">
        <span><i style={{ background: 'var(--up)' }} />panel method</span>
        {polar?.cl_thin && <span><i style={{ background: 'transparent', border: '1.5px dashed var(--muted)' }} />thin-airfoil theory</span>}
        <span><i style={{ background: 'var(--heading)' }} />current angle</span>
        {extraSeries.map((s) => (
          <span key={s.label}><i style={{ background: s.color ?? 'var(--lo)' }} />{s.label}</span>
        ))}
      </div>
      <div className="status err" role="status">{error}</div>
    </section>
  );
}
