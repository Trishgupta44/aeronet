import { useRef, useState } from 'react';
import { optimizeShape } from '../api.js';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';

const f = (v, d = 2) => (v == null ? '–' : v.toFixed(d));
const BOUND_NAMES = { m: 'camber', p: 'camber position', t: 'thickness' };

function axes(g, w, h, m, xs, ys, xLabel, yLabel, flipY) {
  const X = (v) => m.l + ((v - xs[0]) / (xs[1] - xs[0])) * (w - m.l - m.r);
  const Y = (v) => (flipY ? m.t + ((v - ys[0]) / (ys[1] - ys[0])) * (h - m.t - m.b) : h - m.b - ((v - ys[0]) / (ys[1] - ys[0])) * (h - m.t - m.b));
  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  g.font = '12px system-ui';
  g.fillStyle = cssVar('--muted');
  g.strokeStyle = cssVar('--grid');
  g.lineWidth = 1;
  const stepY = ys[1] - ys[0] > 3 ? 1 : 0.5;
  for (let v = Math.ceil(ys[0] / stepY) * stepY; v <= ys[1] + 1e-9; v += stepY) {
    g.beginPath();
    g.moveTo(m.l, Y(v));
    g.lineTo(w - m.r, Y(v));
    g.stroke();
    g.textAlign = 'right';
    g.fillText(v.toFixed(1), m.l - 6, Y(v) + 4);
  }
  g.textAlign = 'center';
  g.fillText(xLabel, (m.l + w - m.r) / 2, h - 6);
  g.save();
  g.translate(12, (m.t + h - m.b) / 2);
  g.rotate(-Math.PI / 2);
  g.fillText(yLabel, 0, 0);
  g.restore();
  return { X, Y };
}

function line(g, pts, X, Y, color, width, dash) {
  g.strokeStyle = color;
  g.lineWidth = width;
  g.setLineDash(dash || []);
  g.beginPath();
  pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
  g.stroke();
  g.setLineDash([]);
}

function ShapePlot({ data }) {
  const ref = useRef(null);
  useCanvasDraw(ref, (w) => Math.max(150, Math.min(220, w * 0.28)), (g, w, h) => {
    const m = { l: 46, r: 14, t: 10, b: 28 };
    const span = (w - m.l - m.r) / 1; // pixels per chord
    const half = (h - m.t - m.b) / 2 / span; // y half-range that keeps the aspect ratio 1:1
    const { X, Y } = axes(g, w, h, m, [0, 1], [-half, half], 'x / chord (same scale in y)', 'y / chord', false);
    [[data.before, '--ref', [6, 4], 1.6], [data.after, '--up', [], 2.2]].forEach(([d, c, dash, wd]) => {
      line(g, d.shape.x.map((x, i) => [x, d.shape.y[i]]).concat([[d.shape.x[0], d.shape.y[0]]]), X, Y, cssVar(c), wd, dash);
    });
  }, [data]);
  return <canvas ref={ref} aria-label="Outline of the starting airfoil and the optimised shape" role="img" />;
}

function CpPlot({ data }) {
  const ref = useRef(null);
  useCanvasDraw(ref, (w) => Math.max(240, Math.min(320, w * 0.4)), (g, w, h) => {
    const m = { l: 46, r: 14, t: 12, b: 34 };
    const all = [...data.before.surface.cp, ...data.after.surface.cp];
    const lo = Math.min(-1, Math.floor((Math.min(...all) - 0.1) * 2) / 2);
    const { X, Y } = axes(g, w, h, m, [0, 1], [lo, 1.5], 'x / chord', 'Cp, Mach 0 (suction is up)', true);
    [[data.before, '--ref', [6, 4], 1.6], [data.after, '--up', [], 2.2]].forEach(([d, c, dash, wd]) => {
      [true, false].forEach((side) => {
        const pts = d.surface.cx.map((x, i) => [x, d.surface.cp[i], d.surface.upper[i]]).filter((p) => p[2] === side).sort((a, b) => a[0] - b[0]);
        line(g, pts, X, Y, cssVar(c), wd, dash);
      });
    });
  }, [data]);
  return <canvas ref={ref} aria-label="Cp distributions of the starting airfoil and the optimised shape" role="img" />;
}

function HistoryPlot({ data }) {
  const ref = useRef(null);
  useCanvasDraw(ref, () => 130, (g, w, h) => {
    const m = { l: 46, r: 14, t: 10, b: 28 };
    const hist = data.history;
    const lo = Math.min(...hist);
    const hi = Math.max(...hist);
    const { X, Y } = axes(g, w, h, m, [0, hist.length - 1], [lo - 0.1, hi + 0.1], 'evaluation', 'best Cp min', false);
    line(g, hist.map((v, i) => [i, v]), X, Y, cssVar('--up'), 2);
  }, [data]);
  return <canvas ref={ref} aria-label="Convergence: best lowest Cp found against evaluations" role="img" />;
}

function Column({ d, title }) {
  return (
    <div className="metric">
      <div className="k">{title}</div>
      <div className="v">{f(d.mach_crit_pg)}</div>
      <div className="n">
        critical Mach (Prandtl–Glauert) · {f(d.mach_crit_kt)} Kármán–Tsien<br />
        angle {f(d.alpha, 1)}° · lowest Cp {f(d.cp_min)} · Cl {f(d.cl, 3)}<br />
        camber {f(d.m * 100, 1)}%{d.m > 0 && <>, position {f(d.p, 2)}c</>}, thickness {f(d.t * 100, 1)}%
      </div>
      <span className="tag tag--measured">computed</span>
    </div>
  );
}

/** "Beat the critical Mach": optimise a NACA 4-digit-family shape for the least suction at a target Cl. Inviscid only. */
export default function OptimizeCard({ code }) {
  const [targetCl, setTargetCl] = useState(0.6);
  const [minT, setMinT] = useState(0.09);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const run = async () => {
    setBusy(true);
    setError('');
    try {
      setData(await optimizeShape({ target_cl: targetCl, min_thickness: minT, code }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card">
      <h2>Beat the critical Mach number</h2>
      <p className="hint">
        <b>Inviscid panel-method optimisation, no drag.</b> Find the NACA 4-digit-family shape (camber, camber position, thickness) with the least suction at a
        given lift, so that the flow stays below the speed of sound up to a higher Mach number. The starting shape is the airfoil selected above (NACA {code}).
      </p>
      <label className="field" htmlFor="opt-cl">Target lift coefficient Cl <span className="val">{targetCl.toFixed(2)}</span></label>
      <input type="range" id="opt-cl" min="0" max="1.2" step="0.05" value={targetCl} onChange={(e) => setTargetCl(+e.target.value)} />
      <label className="field" htmlFor="opt-t" style={{ marginTop: 14 }}>Minimum thickness <span className="val">{(minT * 100).toFixed(0)}% of chord</span></label>
      <input type="range" id="opt-t" min="0.06" max="0.15" step="0.01" value={minT} onChange={(e) => setMinT(+e.target.value)} />
      <div style={{ marginTop: 14 }}>
        <button type="button" className="btn" onClick={run} disabled={busy}>{busy ? 'Optimising…' : 'Optimise'}</button>
      </div>
      {error && <div className="placeholder" role="status" style={{ marginTop: 12 }}>{error}</div>}
      {data && (
        <>
          <div className="metrics">
            <Column d={data.before} title={`Before: ${data.before.name}`} />
            <Column d={data.after} title="After: optimised shape" />
            <div className="metric">
              <div className="k">Gain in critical Mach number</div>
              <div className="v">{data.gain_pg != null ? `+${data.gain_pg.toFixed(2)}` : '–'}</div>
              <div className="n">
                nearest standard airfoil {data.rounded.name}: {f(data.rounded.mach_crit_pg)} (Prandtl–Glauert)<br />
                {data.evaluations} panel solutions in {data.elapsed_s.toFixed(1)} s
              </div>
              <span className="tag tag--measured">computed</span>
            </div>
          </div>
          <ShapePlot data={data} />
          <CpPlot data={data} />
          <HistoryPlot data={data} />
          <div className="legend">
            <span><i style={{ background: 'var(--ref)' }} />before (dashed)</span>
            <span><i style={{ background: 'var(--up)' }} />after</span>
          </div>
          {data.active_bounds.length > 0 && (
            <p className="status">
              The search hit a limit: {data.active_bounds.map((k) => BOUND_NAMES[k]).join(', ')}. Allowed ranges: camber 0 to 6%, camber position 0.2 to 0.7 of the chord,
              thickness {(data.min_thickness * 100).toFixed(0)}% to 18%. A wider range could do better; it would also move away from real, buildable airfoils.
            </p>
          )}
          <p className="src">{data.limits} The optimised shape is not a standard NACA code, so the nearest one is evaluated too. Larger camber far aft gives a strong nose-down pitching moment; that is not limited here.</p>
        </>
      )}
    </section>
  );
}
