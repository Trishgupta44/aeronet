import { useEffect, useRef, useState } from 'react';
import { getTradeoff } from '../api.js';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';

const mesh = (m) => m.replace('x', '×');
const fmtSec = (s) => (s == null ? 'withheld' : s < 1 ? `${(s * 1000).toFixed(1)} ms` : `${s.toFixed(0)} s`);
const fmtRms = (v) => (v == null ? '–' : v.toFixed(4));

/** Everything the chart draws as a point: [{x seconds, y rms, kind, label, detail, ...}] (own time and, for grid sequencing, the total). */
function buildMarks(data) {
  const marks = [];
  if (data.panel) {
    marks.push({ id: 'panel', x: data.panel.seconds, y: data.panel.rms, shape: 'square', filled: true, label: 'panel',
      detail: `Panel method: Cp RMS ${fmtRms(data.panel.rms)} (measured, summary.csv), ${fmtSec(data.panel.seconds)} per solve (measured now on this server)` });
  }
  data.points.forEach((p) => {
    if (p.seconds == null) return;
    const gs = p.grid_sequenced;
    const filled = !p.method.includes('nopos');
    marks.push({ id: p.id, x: p.seconds, y: p.rms, shape: gs ? 'triangle' : 'circle', filled, label: mesh(p.mesh), total: gs ? p.total_seconds : null,
      detail: `${p.label}, ${mesh(p.mesh)}: Cp RMS ${fmtRms(p.rms)}, ${fmtSec(p.seconds)} training${gs && p.total_seconds != null ? ` for this stage, ${fmtSec(p.total_seconds)} including the coarse run it started from` : ''} (measured)` });
  });
  return marks;
}

function drawChart(g, w, h, data, marks, layout) {
  const narrow = w < 430;
  const m = { l: 56, r: narrow ? 78 : 92, t: 14, b: 40 };
  const xs = marks.flatMap((k) => [k.x, k.total].filter((v) => v != null));
  const ys = [...marks.map((k) => k.y), ...data.fvm.map((f) => f.rms)];
  const xlo = Math.pow(10, Math.floor(Math.log10(Math.min(...xs)) - 0.3));
  const xhi = Math.pow(10, Math.ceil(Math.log10(Math.max(...xs)) + 0.1));
  const ylo = Math.pow(10, Math.floor(Math.log10(Math.min(...ys)) * 2) / 2);
  const yhi = Math.pow(10, Math.ceil(Math.log10(Math.max(...ys)) * 2) / 2);
  const X = (v) => m.l + ((Math.log10(v) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo))) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((Math.log10(v) - Math.log10(ylo)) / (Math.log10(yhi) - Math.log10(ylo))) * (h - m.t - m.b);

  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  g.font = '12px system-ui';
  g.strokeStyle = cssVar('--grid');
  g.fillStyle = cssVar('--muted');
  g.lineWidth = 1;
  const decades = Math.floor(Math.log10(xhi)) - Math.ceil(Math.log10(xlo)) + 1;
  const skip = (w - m.l - m.r) / decades < 50 ? 2 : 1; // narrow chart: label every other decade
  for (let e = Math.ceil(Math.log10(xlo)); e <= Math.floor(Math.log10(xhi)); e += 1) {
    const x = X(10 ** e);
    g.beginPath();
    g.moveTo(x, m.t);
    g.lineTo(x, h - m.b);
    g.stroke();
    if ((e - Math.ceil(Math.log10(xlo))) % skip) continue;
    g.textAlign = 'center';
    g.fillText(e < 0 ? `${(10 ** (e + 3)).toFixed(0)} ms` : `${10 ** e} s`, x, h - m.b + 16);
  }
  for (const v of [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]) {
    if (v < ylo || v > yhi) continue;
    g.beginPath();
    g.moveTo(m.l, Y(v));
    g.lineTo(w - m.r, Y(v));
    g.stroke();
    g.textAlign = 'right';
    g.fillText(String(v), m.l - 6, Y(v) + 4);
  }
  g.textAlign = 'center';
  g.fillText('time on this laptop (log scale)', (m.l + w - m.r) / 2, h - 6);
  g.save();
  g.translate(12, (m.t + h - m.b) / 2);
  g.rotate(-Math.PI / 2);
  g.fillText('Cp RMS error vs FVM 800×400 (log scale)', 0, 0);
  g.restore();

  // FVM meshes: error levels only (their times come from other hardware and are quoted, so they are not on the time axis)
  g.save();
  g.setLineDash([6, 4]);
  g.strokeStyle = cssVar('--ref');
  g.fillStyle = cssVar('--ref');
  g.lineWidth = 1.2;
  g.textAlign = 'left';
  data.fvm.forEach((f) => {
    g.beginPath();
    g.moveTo(m.l, Y(f.rms));
    g.lineTo(w - m.r, Y(f.rms));
    g.stroke();
    g.fillText(`FVM ${mesh(f.mesh)}`, w - m.r + 6, Y(f.rms) + 4);
  });
  g.restore();

  const color = (k) => cssVar(k.id === 'panel' ? '--lo' : '--up');
  const placed = [];
  const drawn = []; // label boxes already drawn: a label that would overlap one is left out (hover still gives the numbers)
  marks.forEach((k) => {
    const px = X(k.x);
    const py = Y(k.y);
    if (k.total != null) {
      g.strokeStyle = color(k);
      g.lineWidth = 1.2;
      g.beginPath();
      g.moveTo(px, py);
      g.lineTo(X(k.total), py);
      g.moveTo(X(k.total), py - 5);
      g.lineTo(X(k.total), py + 5);
      g.stroke();
      placed.push({ px: X(k.total), py, detail: `${k.detail}`, k });
    }
    g.strokeStyle = color(k);
    g.fillStyle = k.filled ? color(k) : cssVar('--plot-bg');
    g.lineWidth = 1.8;
    g.beginPath();
    if (k.shape === 'circle') g.arc(px, py, 5.5, 0, 6.2832);
    else if (k.shape === 'square') g.rect(px - 5, py - 5, 10, 10);
    else { g.moveTo(px, py - 6.5); g.lineTo(px + 6, py + 5); g.lineTo(px - 6, py + 5); g.closePath(); }
    g.fill();
    g.stroke();
    const tw = g.measureText(k.label).width;
    const box = { x0: px - tw / 2 - 2, x1: px + tw / 2 + 2, y0: py - 24, y1: py - 8 };
    if (!narrow && !drawn.some((b) => b.x0 < box.x1 && box.x0 < b.x1 && b.y0 < box.y1 && box.y0 < b.y1)) {
      g.fillStyle = cssVar('--muted');
      g.textAlign = 'center';
      g.fillText(k.label, px, py - 11);
      drawn.push(box);
    }
    placed.push({ px, py, detail: k.detail, k });
  });
  layout.current = placed;
}

function Chart({ data }) {
  const ref = useRef(null);
  const layout = useRef([]);
  const [hover, setHover] = useState('');
  const marks = buildMarks(data);

  useCanvasDraw(ref, (w) => Math.max(260, Math.min(360, w * 0.58)), (g, w, h) => drawChart(g, w, h, data, marks, layout), [data]);

  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const near = layout.current.map((p) => ({ p, d: Math.hypot(p.px - x, p.py - y) })).sort((a, b) => a.d - b.d)[0];
    setHover(near && near.d < 16 ? near.p.detail : '');
  };

  return (
    <>
      <canvas ref={ref} aria-label="Cp error against time on this laptop: panel method, PINN runs and FVM error levels" role="img"
        onMouseMove={onMove} onMouseLeave={() => setHover('')} />
      <div className="legend">
        <span><i style={{ background: 'var(--lo)', borderRadius: 0 }} />panel method</span>
        <span><i style={{ background: 'var(--up)' }} />PINN, cold start</span>
        <span>▲ grid-sequenced (left end: fine stage; tick: plus the coarse run)</span>
        <span>hollow = positivity off</span>
        <span>dashed lines = FVM error levels (time not comparable)</span>
      </div>
      <div className="status" role="status" aria-live="polite" style={{ minHeight: '1.4em' }}>{hover || 'Hover a point for its numbers; the table below lists all of them.'}</div>
    </>
  );
}

function gridSequencingNote(data) {
  const cold = data.points.find((p) => p.mesh === '100x50' && p.method === 'pinn');
  const gs = data.points.find((p) => p.mesh === '100x50' && p.method === 'pinn_gs');
  if (!cold || !gs || cold.seconds == null || gs.seconds == null || gs.total_seconds == null) return null;
  return (
    <>
      <b>Grid sequencing, 100×50 (measured):</b> the fine stage took {gs.seconds.toFixed(0)} s against {cold.seconds.toFixed(0)} s from a cold start, for a similar error ({fmtRms(gs.rms)} against{' '}
      {fmtRms(cold.rms)}). It started from a 50×25 run that took {(gs.total_seconds - gs.seconds).toFixed(0)} s; counting that, the total is {gs.total_seconds.toFixed(0)} s, which is{' '}
      {(cold.seconds / gs.total_seconds).toFixed(1)}× less than a cold start. The 50×25 starting run here was trained for many rounds; stopping it earlier would lower its cost but that
      has not been measured.
    </>
  );
}

export default function TradeoffCard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    getTradeoff().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <section className="card card--evidence">
        <p className="eyebrow">E2 · Cost</p>
        <div className="placeholder">The trade-off chart needs the results files, which this server does not have ({error}). Nothing is shown on purpose: there is no fake data on this page.</div>
      </section>
    );
  }
  if (!data) return <section className="card card--evidence"><p className="eyebrow">E2 · Cost</p><div className="status" role="status">Loading results…</div></section>;

  const note = gridSequencingNote(data);
  return (
    <section className="card card--evidence" id="tradeoff">
      <p className="eyebrow">E2 · Cost</p>
      <h3 className="card-title">Accuracy against cost</h3>
      <p className="card-lede">
        NACA 0012, {data.case.alpha_deg}°, Mach {data.case.mach}. Error is the Cp RMS against the {data.reference} solution; time is measured on this laptop
        (PINN training seconds from results/summary.csv, the panel solve timed now). Lower left is better.
      </p>
      <div className="tradeoff-chart"><Chart data={data} /></div>
      {note && <p className="tradeoff-note">{note}</p>}
      <details className="fold fold--inner">
        <summary><span>Show the table</span></summary>
        <div className="fold-body">
          <div className="table-scroll">
            <table className="results-table compact">
              <thead><tr><th>Method</th><th>Mesh</th><th className="num">Time (this laptop)</th><th className="num">Cp RMS</th><th>Label</th></tr></thead>
              <tbody>
                {data.panel && <tr><td>Panel method</td><td>140 panels</td><td className="num mono">{fmtSec(data.panel.seconds)} per solve</td><td className="num mono">{fmtRms(data.panel.rms)}</td><td>measured</td></tr>}
                {data.neuralfoil.available && (
                  <tr><td>NeuralFoil (viscous)</td><td>–</td><td className="num mono">{data.neuralfoil.call_ms.toFixed(1)} ms per call</td><td className="num">no Cp output</td><td title={data.neuralfoil.note}>measured now</td></tr>
                )}
                {data.points.map((p) => (
                  <tr key={p.id}>
                    <td>{p.label}</td>
                    <td className="mono">{mesh(p.mesh)}</td>
                    <td className="num mono" title={p.seconds == null ? p.seconds_note ?? undefined : undefined}>
                      {p.seconds != null ? fmtSec(p.seconds) : 'withheld'}
                      {p.grid_sequenced && p.total_seconds != null && ` (${fmtSec(p.total_seconds)} with the coarse run)`}
                    </td>
                    <td className="num mono">{fmtRms(p.rms)}</td>
                    <td>measured</td>
                  </tr>
                ))}
                {data.fvm.map((f) => (
                  <tr key={f.mesh}>
                    <td>FVM</td>
                    <td className="mono">{mesh(f.mesh)}</td>
                    <td className="num mono">{fmtSec(f.paper_seconds)} <span className="tbd">quoted, other hardware</span></td>
                    <td className="num mono">{fmtRms(f.rms)}</td>
                    <td>error measured, time quoted</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="src">
            Sources: results/summary.csv (E2 and B0 rows). FVM times are quoted from the NNfoil paper (Cao, Song &amp; Zhang), Table 1, which its authors say is not a rigorous
            comparison; they come from other hardware, so they are shown in the table only. NeuralFoil gives Cl and Cd but no pressure distribution, so it has no point on this chart.
          </p>
        </div>
      </details>
    </section>
  );
}
