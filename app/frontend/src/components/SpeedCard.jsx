import { useMemo } from 'react';
import { LOG_SPEED_MAX_MS, LOG_SPEED_MIN_MS, logPos, speedBars } from '../utils/evidence.js';
import { fmtMs, fmtSec } from '../utils/format.js';
import Tag from './Tag.jsx';

// Decades of the log axis: 0.1 ms ... 1000 s (seven decades, equal width).
const TICKS = [[0, '0.1 ms'], [2, '10 ms'], [4, '1 s'], [6, '100 s']];
const GROUPS = [
  { key: 'pinn', title: 'Trained PINN, full field (dark 8 threads, light 1 thread)', test: (b) => b.id.startsWith('pinn') },
  { key: 'tool', title: 'Measured here, no training', test: (b) => b.id === 'panel' || b.id === 'neuralfoil' },
  { key: 'fvm', title: 'Finite-volume solver, quoted from the paper (another machine)', test: (b) => b.id.startsWith('fvm') },
];

function Fill({ b }) {
  const pos = logPos(b.ms);
  const w = `${(pos * 100).toFixed(2)}%`;
  const cls = b.id.startsWith('pinn') ? (b.id.endsWith('-1') ? 'pinn1' : 'pinn8') : b.id === 'panel' ? 'panel' : 'hatched';
  const extra = b.kind === 'viscous' ? 'viscous, not comparable' : b.kind === 'quoted' ? 'quoted' : null;
  const inside = pos > 0.7;
  return (
    <div className={`sbar-line sbar-line--${cls}`}>
      <span className={`sbar-fill sbar-fill--${cls}`} style={{ width: w }} />
      <span className={`sbar-value${inside ? ' sbar-value--in' : ''}`} style={inside ? { left: 6 } : { left: `calc(${w} + 8px)` }}>
        {fmtMs(b.ms)}
        {extra && <em> {extra}</em>}
      </span>
    </div>
  );
}

// One row per tool; the PINN timings of one mesh share a row (8 threads on top, 1 thread below).
function rowsOf(bars) {
  const rows = [];
  bars.forEach((b) => {
    const m = /^pinn-(.+)-(8|1)$/.exec(b.id);
    if (m) {
      const key = `pinn-${m[1]}`;
      let row = rows.find((r) => r.key === key);
      if (!row) {
        row = { key, label: `PINN ${m[1].replace('x', '×')}`, bars: [] };
        rows.push(row);
      }
      row.bars.push(b);
    } else rows.push({ key: b.id, label: b.label, bars: [b] });
  });
  return rows;
}

/**
 * The speed card (experiment E4): horizontal bars on a log axis, one per timing. The numbers come from the E4 rows of results/summary.csv;
 * the PINN 1 thread times are read from the row notes. The panel row of the table is also timed live by the backend on every request.
 */
export default function SpeedCard({ rows, result }) {
  const bars = useMemo(() => speedBars(rows), [rows]);
  if (!bars.length) {
    return (
      <section className="card">
        <p className="eyebrow">E4 · Speed</p>
        <div className="placeholder">The speed rows are not available from this server. Nothing is shown on purpose: there is no fake data on this page.</div>
      </section>
    );
  }
  return (
    <section className="card card--evidence" id="speed">
      <p className="eyebrow">E4 · Speed</p>
      <h3 className="card-title">Milliseconds per query, for one condition</h3>
      <p className="card-lede">
        One query on one laptop (AMD Ryzen 9 6900HS), mean of 100 calls. <b>Not like for like:</b> a PINN answers only the condition it was trained for;
        the panel method needs no training.
      </p>
      <div className="sbars" role="img" aria-label={`Time per query on a log axis from 0.1 ms to 1000 s: ${bars.map((b) => `${b.label} ${fmtMs(b.ms)}`).join('; ')}`}>
        {GROUPS.map((gr) => {
          const items = bars.filter(gr.test);
          if (!items.length) return null;
          return (
            <div className="sbar-group" key={gr.key}>
              <p className="sbar-group-title">{gr.title}</p>
              {rowsOf(items).map((r) => (
                <div className="sbar" key={r.key}>
                  <div className="sbar-label">{r.label}</div>
                  <div className="sbar-track">{r.bars.map((b) => <Fill key={b.id} b={b} />)}</div>
                </div>
              ))}
            </div>
          );
        })}
        <div className="sbar sbar--axis" aria-hidden="true">
          <div className="sbar-label" />
          <div className="sbar-ticks">
            {TICKS.map(([k, t]) => (
              <span key={t} style={{ left: `${(k / 7) * 100}%` }}>{t}</span>
            ))}
          </div>
        </div>
      </div>
      <p className="hint">
        Log scale, {LOG_SPEED_MIN_MS} ms to {LOG_SPEED_MAX_MS / 1000} s. Hatched = not comparable: NeuralFoil is viscous; the FVM times are the paper&rsquo;s, on other hardware.
      </p>
      <details className="fold fold--inner">
        <summary><span>Show the exact numbers</span></summary>
        <div className="fold-body">
          <div className="table-scroll">
            <table className="results-table compact">
              <thead><tr><th>What is timed</th><th className="num">Per query</th><th className="num">Training, once</th><th>Source</th><th>Note</th></tr></thead>
              <tbody>
                {result?.solve_ms != null && (
                  <tr>
                    <td className="live">Panel method, this app, just now</td>
                    <td className="num mono live">{fmtMs(result.solve_ms)}</td>
                    <td className="num mono">–</td>
                    <td><Tag kind="measured">measured now</Tag></td>
                    <td className="note">timed by the server on the last solve</td>
                  </tr>
                )}
                {bars.map((b) => (
                  <tr key={b.id}>
                    <td>{b.label}</td>
                    <td className="num mono">{fmtMs(b.ms)}</td>
                    <td className="num mono">{b.id.startsWith('pinn') ? (b.trainS != null ? fmtSec(b.trainS) : <span className="tbd" title="the laptop slept during this run">withheld</span>) : '–'}</td>
                    <td><Tag kind={b.kind === 'quoted' ? 'quoted' : 'measured'} /></td>
                    <td className="note" title={b.note}>{b.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="src">Source: results/summary.csv, experiment E4; conditions and spread in results/e4_speed.md. Training time of grid-sequenced runs excludes the run that supplied the weights; the 200×100 time is withheld (the laptop slept). Quoted = copied from arXiv 2401.08705 (Cao, Song and Zhang) and not re-run here.</p>
        </div>
      </details>
    </section>
  );
}
