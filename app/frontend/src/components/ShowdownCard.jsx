import { useEffect, useRef, useState } from 'react';
import { getShowdown } from '../api.js';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';

const fmt = (v, d = 4) => (v == null ? '–' : v.toFixed(d));

// One wall-Cp curve per surface (upper, lower), sorted by x, in one colour.
function strokeCurve(g, curve, X, Y, color, width, dash) {
  g.strokeStyle = color;
  g.lineWidth = width;
  g.setLineDash(dash || []);
  [true, false].forEach((side) => {
    const pts = curve.x.map((x, i) => [x, curve.cp[i], curve.upper[i]]).filter((p) => p[2] === side);
    g.beginPath();
    pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
    g.stroke();
  });
  g.setLineDash([]);
}

function drawShowdown(g, w, h, reference, fvm, pinn) {
  const m = { l: 46, r: 14, t: 12, b: 34 };
  const all = [...reference.cp, ...(fvm ? fvm.curve.cp : []), ...(pinn?.curve ? pinn.curve.cp : [])];
  const lo = Math.min(-1, Math.floor((Math.min(...all) - 0.1) * 2) / 2);
  const hi = Math.max(1.5, Math.ceil(Math.max(...all) * 2) / 2);
  const X = (x) => m.l + x * (w - m.l - m.r);
  const Y = (v) => m.t + ((v - lo) / (hi - lo)) * (h - m.t - m.b); // suction is up

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
  g.fillText('Cp (suction is up)', 0, 0);
  g.restore();

  strokeCurve(g, reference, X, Y, cssVar('--ref'), 1.4);
  if (fvm) strokeCurve(g, fvm.curve, X, Y, cssVar('--lo'), 1.8, [6, 4]);
  if (pinn?.curve) strokeCurve(g, pinn.curve, X, Y, cssVar('--up'), 2);
}

function Block({ title, rms, max, cl, extra, tag }) {
  return (
    <div className="metric">
      <div className="k">{title}</div>
      <div className="v">{fmt(rms)}</div>
      <div className="n">
        Cp RMS error · max {fmt(max, 3)} · Cl {fmt(cl, 3)}
        {extra && <><br />{extra}</>}
      </div>
      {tag && <span className="tag tag--measured">{tag}</span>}
    </div>
  );
}

export default function ShowdownCard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    getShowdown().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <section className="card">
        <h2>Same-mesh showdown</h2>
        <div className="placeholder">Trained-case results are not available from this server ({error}). Nothing is shown on purpose: there is no fake data on this page.</div>
      </section>
    );
  }
  if (!data) return <section className="card"><h2>Same-mesh showdown</h2><div className="status" role="status">Loading results…</div></section>;
  return <ShowdownBody data={data} />; // separate component so the canvas hook mounts together with the canvas
}

function ShowdownBody({ data }) {
  const [mesh, setMesh] = useState('100x50');
  const [method, setMethod] = useState(null);
  const ref = useRef(null);

  const entry = data.meshes.find((x) => x.mesh === mesh);
  const variants = entry?.pinn ?? [];
  const pinn = variants.find((v) => v.method === method) ?? variants[0];
  const ready = pinn?.status === 'done';

  useCanvasDraw(ref, (w) => Math.max(260, Math.min(380, w * 0.45)), (g, w, h) => {
    drawShowdown(g, w, h, data.reference.curve, entry.fvm, ready ? pinn : null);
  }, [data, mesh, pinn]);

  return (
    <section className="card">
      <h2>Same-mesh showdown: finite-volume solver vs PINN</h2>
      <p className="hint">
        NACA 0012, {data.case.alpha_deg}°, Mach {data.case.mach}. Both solvers run on the <b>same mesh</b>; both errors are measured against the
        finer {data.reference.label} solution. PINN method after Cao, Song &amp; Zhang (NNfoil).
      </p>
      <div className="seg" role="group" aria-label="Mesh">
        {data.meshes.map((x) => (
          <button key={x.mesh} type="button" aria-pressed={x.mesh === mesh} onClick={() => { setMesh(x.mesh); setMethod(null); }}>
            {x.mesh.replace('x', '×')}
          </button>
        ))}
      </div>
      {variants.length > 1 && (
        <div className="seg" role="group" aria-label="PINN variant" style={{ marginTop: 6 }}>
          {variants.map((v) => (
            <button key={v.method} type="button" aria-pressed={v.method === pinn.method} onClick={() => setMethod(v.method)}>{v.label}</button>
          ))}
        </div>
      )}
      <div><canvas ref={ref} aria-label="Wall Cp: reference, finite-volume and PINN" role="img" /></div>
      <div className="legend">
        <span><i style={{ background: 'var(--ref)' }} />{data.reference.label} (reference)</span>
        {entry.fvm && <span><i style={{ background: 'var(--lo)' }} />FVM {mesh.replace('x', '×')}</span>}
        {ready && <span><i style={{ background: 'var(--up)' }} />PINN {mesh.replace('x', '×')}, final round</span>}
      </div>
      <div className="metrics">
        {entry.fvm ? <Block title={`FVM ${mesh.replace('x', '×')}`} {...entry.fvm} tag="measured" /> : <div className="placeholder">No FVM row for this mesh.</div>}
        {ready && (
          <Block
            title={pinn.label}
            {...pinn}
            extra={`${pinn.train_s != null ? `${pinn.train_s.toFixed(0)} s training · ` : ''}${pinn.rounds ?? '?'} rounds, ${pinn.failed_rounds ?? '?'} failed`}
            tag="measured"
          />
        )}
        {pinn && !ready && (
          <div className="metric">
            <div className="k">{pinn.label}</div>
            <div className="v">pending</div>
            <div className="n">
              {pinn.rounds_done} of {pinn.rounds_target ?? '?'} rounds logged. No curve or error is shown until the run finishes and is collected into results/summary.csv.
            </div>
          </div>
        )}
        {!pinn && <div className="placeholder">No PINN run for this mesh yet.</div>}
      </div>
      {entry.fvm && ready && (
        <div className="status" role="status">
          On the {mesh.replace('x', '×')} mesh the PINN error is <b>{(entry.fvm.rms / pinn.rms).toFixed(1)}× {pinn.rms < entry.fvm.rms ? 'smaller' : 'larger'}</b> than the
          finite-volume error (computed from the two measured numbers above).
        </div>
      )}
      <p className="src">
        Source: results/summary.csv (experiments B0 and E2) and the final-round Cp of each run; the final round is used, never the best round (choosing it would use the reference answer).
        Error = {data.error_definition}.
        {mesh === '50x25' && ' On 50×25 the PINN drifts in late rounds: its error was lowest around rounds 3 to 7 and rose afterwards (docs/nnfoil_improvements.md, section 3c).'}
      </p>
    </section>
  );
}
