import { useEffect, useRef, useState } from 'react';
import { getViscous } from '../api.js';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';

const RE_CHOICES = [
  [1e5, '100 k'],
  [5e5, '500 k'],
  [1e6, '1 M'],
  [3e6, '3 M'],
  [1e7, '10 M'],
];
const fmt = (v, d = 3) => (v == null ? '–' : v.toFixed(d));

function drawPolar(g, w, h, d) {
  const { alpha, cl, cl_inviscid: inv } = d.polar;
  const m = { l: 46, r: 14, t: 12, b: 34 };
  const all = [...cl, ...inv];
  const lo = Math.floor(Math.min(...all) * 2) / 2;
  const hi = Math.ceil(Math.max(...all) * 2) / 2;
  const a0 = alpha[0];
  const a1 = alpha[alpha.length - 1];
  const X = (a) => m.l + ((a - a0) / (a1 - a0)) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((v - lo) / (hi - lo)) * (h - m.t - m.b);

  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  g.font = '12px system-ui';
  g.fillStyle = cssVar('--muted');
  g.strokeStyle = cssVar('--grid');
  g.lineWidth = 1;
  for (let v = lo; v <= hi + 1e-9; v += 0.5) {
    g.beginPath();
    g.moveTo(m.l, Y(v));
    g.lineTo(w - m.r, Y(v));
    g.stroke();
    g.textAlign = 'right';
    g.fillText(v.toFixed(1), m.l - 6, Y(v) + 4);
  }
  for (let a = Math.ceil(a0 / 4) * 4; a <= a1; a += 4) {
    g.beginPath();
    g.moveTo(X(a), m.t);
    g.lineTo(X(a), h - m.b);
    g.stroke();
    g.textAlign = 'center';
    g.fillText(`${a}°`, X(a), h - m.b + 16);
  }
  g.fillText('angle of attack', (m.l + w - m.r) / 2, h - 6);
  g.save();
  g.translate(12, (m.t + h - m.b) / 2);
  g.rotate(-Math.PI / 2);
  g.fillText('Cl', 0, 0);
  g.restore();

  const line = (ys, color, dash, width) => {
    g.strokeStyle = color;
    g.lineWidth = width;
    g.setLineDash(dash);
    g.beginPath();
    alpha.forEach((a, i) => (i ? g.lineTo(X(a), Y(ys[i])) : g.moveTo(X(a), Y(ys[i]))));
    g.stroke();
    g.setLineDash([]);
  };
  line(inv, cssVar('--ref'), [6, 4], 1.6);
  line(cl, cssVar('--up'), [], 2.2);
  [[d.inviscid.cl, '--ref'], [d.cl, '--up']].forEach(([v, c]) => {
    g.fillStyle = cssVar(c);
    g.beginPath();
    g.arc(X(d.alpha), Y(v), 4.5, 0, 6.2832);
    g.fill();
  });
}

// Its own component so the canvas hook mounts together with the canvas (the canvas only exists once there is data).
function PolarCanvas({ data }) {
  const ref = useRef(null);
  useCanvasDraw(ref, (w) => Math.max(240, Math.min(320, w * 0.4)), (g, w, h) => drawPolar(g, w, h, data), [data]);
  return <div><canvas ref={ref} aria-label="Lift coefficient versus angle of attack, inviscid and viscous" role="img" /></div>;
}

/** Viscous reality check for NACA airfoils: NeuralFoil estimate next to the inviscid panel result. Not shown for other shapes. */
export default function ViscousCard({ code, alpha }) {
  const [re, setRe] = useState(1e6);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const ctl = new AbortController();
    const t = setTimeout(() => {
      getViscous({ code, alpha, re }, ctl.signal)
        .then((d) => { setData(d); setError(''); })
        .catch((e) => { if (e.name !== 'AbortError') setError(e.message); });
    }, 300);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [code, alpha, re]);


  return (
    <section className="card">
      <h2>Viscous reality check: what the inviscid model leaves out</h2>
      <p className="hint">
        The panel solver ignores viscosity, so it predicts no drag. NeuralFoil (Peter Sharpe, MIT licence), a network trained on XFOIL results,
        estimates the viscous answer at a chosen Reynolds number. It is shown for comparison only and is not part of the PINN work.
      </p>
      <div className="seg" role="group" aria-label="Reynolds number">
        {RE_CHOICES.map(([v, label]) => (
          <button key={v} type="button" aria-pressed={v === re} onClick={() => setRe(v)}>Re {label}</button>
        ))}
      </div>
      {error && <div className="placeholder" role="status" style={{ marginTop: 12 }}>{error} Nothing is shown on purpose: there is no fake data on this page.</div>}
      {!error && !data && <div className="status" role="status">Loading…</div>}
      {data && (
        <>
          <div className="metrics">
            <div className="metric">
              <div className="k">Inviscid panel method, Mach 0</div>
              <div className="v">Cl {fmt(data.inviscid.cl)}</div>
              <div className="n">Cd {fmt(data.inviscid.cd, 4)} (zero in theory: numerical error only)</div>
              <span className="tag tag--measured">computed live</span>
            </div>
            <div className="metric">
              <div className="k">{data.label}, Re {RE_CHOICES.find((c) => c[0] === data.re)?.[1] ?? data.re}</div>
              <div className="v">Cl {fmt(data.cl)}</div>
              <div className="n">Cd {fmt(data.cd, 4)} · Cl/Cd {fmt(data.cl_cd, 1)}</div>
              <div className="n">NeuralFoil's own confidence {fmt(data.confidence, 2)} (1 = model applies well)</div>
              <span className="tag">estimate</span>
            </div>
            <div className="metric">
              <div className="k">NeuralFoil time per call</div>
              <div className="v">{data.call_ms.toFixed(1)} ms</div>
              <div className="n">mean of 5 calls on this server, {data.model_size} model, NeuralFoil {data.version ?? ''}</div>
              <span className="tag tag--measured">measured now</span>
            </div>
          </div>
          <PolarCanvas data={data} />
          <div className="legend">
            <span><i style={{ background: 'var(--ref)' }} />inviscid panel method</span>
            <span><i style={{ background: 'var(--up)' }} />NeuralFoil, Re {RE_CHOICES.find((c) => c[0] === data.re)?.[1] ?? data.re}</span>
            <span>dots: the angle selected above</span>
          </div>
          <p className="status">
            Why the inviscid drag is about 0: in steady, inviscid, attached flow a body feels no drag (d'Alembert's paradox). Real drag comes from
            the boundary layer: skin friction and the pressure change it causes. For this case NeuralFoil estimates a drag of {fmt(data.cd, 4)} and a lift of{' '}
            {fmt(data.cl)} against {fmt(data.inviscid.cl)} inviscid ({data.cl < data.inviscid.cl ? 'lower' : 'higher'}; the sign of the lift change depends on the case, so compare the chart above
            rather than expecting a fixed direction). NeuralFoil has no Mach input, so this comparison is at Mach 0.
          </p>
        </>
      )}
    </section>
  );
}
