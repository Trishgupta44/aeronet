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
  // Cp* (where the local flow becomes sonic): dashed line, drawn only when it falls inside the plot range.
  const star = result.validity?.cp_crit;
  if (star != null && star >= lo && star <= hi) {
    g.strokeStyle = cssVar('--flow-stagnation');
    g.lineWidth = 1.4;
    g.setLineDash([7, 5]);
    g.beginPath();
    g.moveTo(m.l, Y(star));
    g.lineTo(w - m.r, Y(star));
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = cssVar('--flow-stagnation');
    g.textAlign = 'right';
    g.fillText(`Cp* = ${star.toFixed(2)} (sonic)`, w - m.r - 4, Y(star) - 5);
  }
}

const f2 = (v) => (v == null ? '–' : v.toFixed(2));

/** Validity of the inviscid subsonic model, from the panel solution (all numbers computed by the backend). */
function ValidityNote({ validity, mach }) {
  if (!validity) return null;
  const { supersonic, cp_min: cpMin, cp_crit: star, mach_crit_pg: pg, mach_crit_kt: kt } = validity;
  return (
    <div className="validity">
      {supersonic && (
        <div className="notice error-banner" role="alert">
          <b>Local flow is supersonic here: the inviscid subsonic model is not valid.</b> At Mach {mach.toFixed(2)} the lowest Cp ({f2(cpMin)}) is below
          the sonic value Cp* ({f2(star)}). Expect a shock in the real flow; treat the Cl and Cp shown here as unreliable.
        </div>
      )}
      <p className="hint">
        <span className="tag tag--measured">computed</span>{' '}
        Lowest Cp {f2(cpMin)}{star != null && <>, sonic Cp* {f2(star)} at Mach {mach.toFixed(2)}</>}. Critical Mach number for this shape and angle:{' '}
        <b>{pg != null ? pg.toFixed(2) : 'above 0.99'}</b> with the Prandtl–Glauert rule (the one this solver uses)
        {kt != null && <>, <b>{kt.toFixed(2)}</b> with the Kármán–Tsien rule</>}.
        Prandtl–Glauert gives the weakest growth of suction with Mach number, so its critical Mach number is the optimistic one; the true value
        is probably closer to the lower number or below it.
      </p>
    </div>
  );
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
        {result?.validity?.cp_crit != null && <span><i style={{ background: 'var(--flow-stagnation)', borderRadius: 0, height: 2, verticalAlign: 3 }} />sonic Cp* (dashed)</span>}
      </div>
      {result && <ValidityNote validity={result.validity} mach={result.mach} />}
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
