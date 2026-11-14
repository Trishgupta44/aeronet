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

// Shaded min-max band of the final-round Cp over the seeds (one polygon per surface).
function fillBand(g, band, X, Y, color) {
  g.save();
  g.fillStyle = color;
  g.globalAlpha = 0.2;
  [true, false].forEach((side) => {
    const idx = band.x.map((_, i) => i).filter((i) => band.upper[i] === side);
    if (idx.length < 2) return;
    g.beginPath();
    idx.forEach((i, k) => (k ? g.lineTo(X(band.x[i]), Y(band.cp_max[i])) : g.moveTo(X(band.x[i]), Y(band.cp_max[i]))));
    [...idx].reverse().forEach((i) => g.lineTo(X(band.x[i]), Y(band.cp_min[i])));
    g.closePath();
    g.fill();
  });
  g.restore();
}

function drawShowdown(g, w, h, reference, fvm, pinn) {
  const m = { l: 46, r: 14, t: 12, b: 34 };
  const band = pinn?.seeds?.band;
  // `reference` and `fvm.curve` are null on public deployments (NNfoil's FVM curves are not served there)
  const all = [...(reference ? reference.cp : []), ...(fvm?.curve ? fvm.curve.cp : []), ...(pinn?.curve ? pinn.curve.cp : []), ...(band ? [...band.cp_min, ...band.cp_max] : []), -1, 1];
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

  if (band) fillBand(g, band, X, Y, cssVar('--up'));
  if (reference) strokeCurve(g, reference, X, Y, cssVar('--ref'), 1.4);
  if (fvm?.curve) strokeCurve(g, fvm.curve, X, Y, cssVar('--lo'), 1.8, [6, 4]);
  if (pinn?.curve) strokeCurve(g, pinn.curve, X, Y, cssVar('--up'), 2);
}

const range = (r, d) => (r ? `${r.min.toFixed(d)} to ${r.max.toFixed(d)}` : '–');

// Facts only, no traffic light. Every line carries its source.
// TODO: the off-grid residual (a reference-free check for late-round drift) is deliberately NOT shown yet: it is exploratory
// (one point per run, mixed meshes). See docs/nnfoil_improvements.md, section 3c, on the main checkout.
function Evidence({ pinn, mesh, mach }) {
  const s = pinn.seeds;
  const v = pinn.validity;
  return (
    <div className="evidence">
      <h3>Evidence for this result</h3>
      <ul className="plain">
        <li>
          <span className="tag tag--measured">measured</span> <b>Run:</b> {mesh.replace('x', '×')} mesh, {pinn.rounds ?? '?'} rounds, {pinn.failed_rounds ?? '?'} failed
          rounds, {pinn.train_s != null ? `${pinn.train_s.toFixed(0)} s training` : `training time withheld (${pinn.train_s_note ?? 'no time recorded'})`}{' '}
          (results/summary.csv, run {pinn.run_id}).
        </li>
        <li>
          <span className="tag tag--measured">measured</span> <b>Seed spread:</b>{' '}
          {s ? (
            <>
              n = {s.n} seeds{s.n_without_result > 0 && <>, {s.n_without_result} without a result (collapsed or failed)</>}. Cl {range(s.cl, 3)}, Cp RMS{' '}
              {range(s.rms, 4)} ({s.source}). {s.rounds_differ && 'The seeds were trained for different numbers of rounds. '}
              {s.band && <>Shaded band: lowest and highest Cp at each wall point over the {s.band.n_curves} seeds that gave a result.</>}
            </>
          ) : (
            'no seed-to-seed data for this variant, so no spread is claimed.'
          )}
        </li>
        <li>
          <span className="tag tag--measured">computed</span> <b>Validity:</b> the PINN's lowest wall Cp is {v.cp_min.toFixed(2)}; the sonic value at Mach {mach} is{' '}
          {v.cp_crit.toFixed(2)}. {v.supersonic ? 'The flow reaches Mach 1 locally, so the subsonic model is not valid.' : 'The flow stays subsonic, so the subsonic model applies.'}
        </li>
        <li>
          <b>Scope:</b> inviscid Euler flow (no boundary layer or separation). Drag is not predicted. Cars and other bluff bodies are only idealized flow, never real drag.
        </li>
      </ul>
    </div>
  );
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
        {data.reference.curve && <span><i style={{ background: 'var(--ref)' }} />{data.reference.label} (reference)</span>}
        {entry.fvm?.curve && <span><i style={{ background: 'var(--lo)' }} />FVM {mesh.replace('x', '×')}</span>}
        {ready && <span><i style={{ background: 'var(--up)' }} />PINN {mesh.replace('x', '×')}, final round</span>}
        {ready && pinn.seeds?.band && <span><i style={{ background: 'var(--up)', opacity: 0.25, borderRadius: 0 }} />min-max over {pinn.seeds.band.n_curves} seeds</span>}
      </div>
      <div className="metrics">
        {entry.fvm ? <Block title={`FVM ${mesh.replace('x', '×')}`} {...entry.fvm} tag="measured" /> : <div className="placeholder">No FVM row for this mesh.</div>}
        {ready && (
          <Block
            title={pinn.label}
            {...pinn}
            extra={`${pinn.train_s != null ? `${pinn.train_s.toFixed(0)} s training · ` : 'training time withheld · '}${pinn.rounds ?? '?'} rounds, ${pinn.failed_rounds ?? '?'} failed`}
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
      {ready && <Evidence pinn={pinn} mesh={mesh} mach={data.case.mach} />}
      {data.fvm_curves_withheld && (
        <p className="hint">
          The finite-volume pressure curves are not shown in this deployment: the NNfoil data they come from has no licence for redistribution. The errors above
          were computed against them and are real; the PINN curves are this project's own results.
        </p>
      )}
      <p className="src">
        Source: results/summary.csv (experiments B0 and E2) and the final-round Cp of each run; the final round is used, never the best round (choosing it would use the reference answer).
        Error = {data.error_definition}.
        {mesh === '50x25' && ' On 50×25 the PINN drifts in late rounds: its error was lowest around rounds 3 to 7 and rose afterwards (measured, in docs/nnfoil_improvements.md, section 3c).'}
      </p>
    </section>
  );
}
