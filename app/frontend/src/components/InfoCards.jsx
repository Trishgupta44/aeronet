// Results card, speed/accuracy table, PINN training history and the project pipeline card.
import { fmt, fmtMs, fmtSec, meshLabel } from '../utils/format.js';
import { findRow } from '../utils/pinnCase.js';
import LineChart from './LineChart.jsx';
import Tag from './Tag.jsx';


export function ResultsCard({ result, idp = '', pinned = null, onPin, onUnpin, onPng, onCsv }) {
  const dash = '–';
  const custom = result?.code === 'custom';
  const noDrag = result?.ground_y != null; // a car on a road: inviscid flow has no drag, so no number is shown
  return (
    <section className="card">
      <div className="card-head">
        <h2>Results</h2>
        {onPin && (
          <div className="actions" role="group" aria-label="Pin and export">
            {pinned ? (
              <button type="button" className="btn" onClick={onUnpin}>Unpin</button>
            ) : (
              <button type="button" className="btn" onClick={onPin} disabled={!result}>Pin result</button>
            )}
            <button type="button" className="btn" onClick={onPng} disabled={!result}>Download PNG</button>
            <button type="button" className="btn" onClick={onCsv} disabled={!result}>Download CSV</button>
          </div>
        )}
      </div>
      <div className="metrics">
        <div className="metric"><div className="k">Lift coefficient Cl</div><div className="v" id={`${idp}mCl`}>{result ? fmt(result.cl, 3) : dash}</div><div className="n">from surface pressure</div></div>
        <div className="metric">
          <div className="k">Thin-airfoil theory</div>
          <div className="v" id={`${idp}mThin`}>{result && result.cl_thin != null ? result.cl_thin.toFixed(3) : custom ? 'n/a' : dash}</div>
          <div className="n">{custom ? 'not an airfoil code' : 'quick sanity check'}</div>
        </div>
        <div className="metric">
          <div className="k">Drag Cd {noDrag ? '' : '(error only)'}</div>
          <div className="v" id={`${idp}mCd`}>{result ? (noDrag ? 'not shown' : fmt(result.cd, 4)) : dash}</div>
          <div className="n">{noDrag ? 'inviscid flow has no drag' : 'ideal value is 0'}</div>
        </div>
        <div className="metric"><div className="k">Solve time</div><div className="v" id={`${idp}mMs`}>{result ? `${result.solve_ms.toFixed(1)} ms` : dash}</div><div className="n">measured on the server</div></div>
      </div>
      {pinned && result && (
        <p className="delta" id={`${idp}delta`}>
          Pinned: <b>{pinned.label}</b>, Cl {pinned.cl.toFixed(3)}. Now: Cl {result.cl.toFixed(3)}, so <b>ΔCl = {(result.cl - pinned.cl >= 0 ? '+' : '−') + Math.abs(result.cl - pinned.cl).toFixed(3)}</b>.
        </p>
      )}
    </section>
  );
}

const dash = '–';

// The speed and accuracy table. Every number comes from a row of results/summary.csv (E4 = speed, E2 = Cp error against FVM 800x400),
// passed in as `rows`; the live panel row is timed by the backend on every request. Without the rows a cell says so: nothing is estimated.
export function ComparisonCard({ result, rows }) {
  const loaded = Array.isArray(rows) && rows.length > 0;
  const e4 = (match) => findRow(rows, { experiment: 'E4', ...match });
  const e2 = (match) => findRow(rows, { experiment: 'E2', ...match });
  const na = <span className="tbd">{loaded ? dash : 'results not loaded'}</span>;
  const pinnRows = [
    { method: 'pinn', mesh: '50x25', label: 'NNfoil PINN, 50×25 (positivity on)' },
    { method: 'pinn', mesh: '100x50', label: 'NNfoil PINN, 100×50 (cold start)' },
    { method: 'pinn_gs', mesh: '100x50', label: 'NNfoil PINN, 100×50 (grid-sequenced)', note: 'training time excludes the 50×25 run that supplied the weights' },
    { method: 'pinn_gs_nopos', mesh: '200x100', label: 'NNfoil PINN, 200×100 (grid-sequenced)', note: 'training time withheld: the laptop slept during this run' },
  ];
  const panel = e4({ method: 'panel' });
  const panelAcc = e2({ method: 'panel' });
  const fvm = ['50x25', '100x50', '200x100', '400x200'].map((mesh) => e4({ method: 'fvm', mesh })).filter(Boolean);
  const fvmAcc = ['50x25', '400x200'].map((mesh) => e2({ method: 'fvm', mesh }));
  const nf = e4({ method: 'neuralfoil' });
  return (
    <section className="card">
      <h2>Speed and accuracy comparison</h2>
      <p className="hint" style={{ marginTop: 0 }}>
        NACA 0012, 4°, Mach 0.4. Accuracy is the Cp RMS against the FVM 800×400 solution (final round for the PINN, never the best round). Times
        are the mean of 100 calls on one laptop (AMD Ryzen 9 6900HS, 8 threads, nothing else running). <b>This is not a like-for-like comparison:</b>{' '}
        a trained network answers one condition and must be retrained for any other, while the panel method needs no training.
      </p>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Method</th><th>Setup</th><th>Per new query</th><th>Cp RMS</th><th>Physics</th><th>Data needed</th></tr></thead>
          <tbody>
            <tr>
              <td className="live">Panel method (this app)</td>
              <td>none</td>
              <td>
                <span className="live" id="tMs">{result ? `${result.solve_ms.toFixed(1)} ms` : dash}</span> <Tag kind="measured">now</Tag>
                {panel && <><br />{fmtMs(panel.infer_ms)} solve only <Tag kind="measured" /></>}
              </td>
              <td>{panelAcc ? <>{fmt(panelAcc.rms_cp, 4)} <Tag kind="measured" /></> : na}</td>
              <td>potential flow + Prandtl–Glauert</td>
              <td>none</td>
            </tr>
            {pinnRows.map((p) => {
              const t = e4({ method: p.method, mesh: p.mesh });
              const a = e2({ method: p.method, mesh: p.mesh });
              return (
                <tr key={p.label}>
                  <td>{p.label}</td>
                  <td>
                    {t ? (t.train_s != null ? <>{fmtSec(t.train_s)} training <Tag kind="measured" /></> : <span className="tbd">withheld</span>) : na}
                    {p.note && t && <><br /><span className="hint">{p.note}</span></>}
                  </td>
                  <td>{t ? <>{fmtMs(t.infer_ms)} full field <Tag kind="measured" /><br /><span className="hint">only the trained condition</span></> : na}</td>
                  <td>{a ? <>{fmt(a.rms_cp, 4)} <Tag kind="measured" /></> : na}</td>
                  <td>Euler equations</td>
                  <td>none</td>
                </tr>
              );
            })}
            <tr>
              <td>FVM (reference solver)</td>
              <td>none</td>
              <td>
                {fvm.length ? (
                  <>
                    {fmtMs(fvm[0].infer_ms)} to {fmtMs(fvm[fvm.length - 1].infer_ms)} per solve, {meshLabel(fvm[0].mesh)} to {meshLabel(fvm[fvm.length - 1].mesh)} <Tag kind="quoted" />
                    <br /><span className="hint">paper, Table 1, the authors' machine</span>
                  </>
                ) : na}
              </td>
              <td>{fvmAcc[0] && fvmAcc[1] ? <>{fmt(fvmAcc[0].rms_cp, 4)} (50×25) to {fmt(fvmAcc[1].rms_cp, 4)} (400×200) <Tag kind="measured" /></> : na}</td>
              <td>Euler equations</td>
              <td>none</td>
            </tr>
            <tr>
              <td>NeuralFoil <span className="tbd">(viscous, not comparable)</span></td>
              <td>pre-trained</td>
              <td>{nf ? <>{fmtMs(nf.infer_ms)} per call <Tag kind="measured" /><br /><span className="hint">Re 6e6, Cl {fmt(nf.cl, 4)}: a different problem</span></> : na}</td>
              <td><span className="tbd">not computed</span></td>
              <td>viscous, learned</td>
              <td>7.9 million XFoil runs <Tag kind="quoted" /></td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="src">Source: results/summary.csv, experiments E4 (speed) and E2 (error); full conditions in results/e4_speed.md. Quoted = copied from arXiv 2401.08705 (Cao, Song and Zhang) and not re-run here.</p>
    </section>
  );
}

/**
 * The PINN training history: loss, Cl and (when there is a reference) the Cp error of every logged round. `round` is the round the
 * replay shows now (a marker on each chart). Without a log the card says so instead of drawing anything.
 */
export function TrainingCard({ log, clRef = null, rms = null, round = null }) {
  if (!log || !log.round || !log.round.length) {
    return (
      <section className="card">
        <h2>Training history</h2>
        <div className="placeholder">
          Pick a trained case and its training log appears here.<br />
          Nothing is shown otherwise on purpose: there is no fake data on this page.
        </div>
      </section>
    );
  }
  const lossSeries = [
    { ys: log.loss, color: '--heading', label: 'total loss' },
    { ys: log.loss_f, color: '--up', dash: [5, 3], label: 'equation residual', width: 1.4 },
    { ys: log.loss_b, color: '--lo', dash: [2, 3], label: 'boundary', width: 1.4 },
  ];
  const clSeries = [{ ys: log.cl, color: '--heading', label: 'Cl of the round' }];
  const hasRms = rms && rms.values.some((v) => v != null);
  const rmsSeries = hasRms ? [{ ys: rms.values, color: '--lo', label: 'Cp RMS vs reference' }] : [];
  return (
    <section className="card">
      <h2>Training history</h2>
      <p className="hint" style={{ marginTop: 0 }}>
        <Tag kind="measured" /> From the training log of this run (one point per optimiser round). The dotted line marks the round shown in the replay above.
        A falling loss does not mean a better answer: compare the curves.
      </p>
      <div className="charts3">
        <LineChart title="Loss (log scale)" xs={log.round} series={lossSeries} logY cursor={round} />
        <LineChart
          title="Lift coefficient Cl"
          xs={log.round}
          series={clSeries}
          cursor={round}
          hline={clRef != null ? { y: clRef, label: `reference ${clRef.toFixed(3)} (FVM 800×400)` } : null}
        />
        {hasRms ? (
          <LineChart title="Cp RMS against FVM 800×400 (log scale)" xs={rms.rounds} series={rmsSeries} logY cursor={round}
            note="Computed here from each round's wall Cp, with the same definition as the exported RMS." />
        ) : (
          <div className="placeholder">No reference solution for this case, so no error curve.</div>
        )}
      </div>
    </section>
  );
}

export function PipelineCard() {
  return (
    <section className="card">
      <h2>Project pipeline and what is real in this prototype</h2>
      <ol className="steps">
        <li className="done"><b>Demo ✓</b>layout, controls, Cp field, Cl checks, draw-any-shape</li>
        <li className="done"><b>Verify ✓</b>NNfoil code reviewed and fixed</li>
        <li className="done"><b>Train ✓</b>NNfoil on this laptop's CPU: 14 exported runs, 50×25 to 200×100 meshes</li>
        <li className="done"><b>Validate ✓</b>PINN against FVM error tables, speed table, stop-rule test</li>
      </ol>
      <ul className="plain">
        <li><b>Real now:</b> airfoil geometry, panel solver (Python backend), Cp field and curve, Cl from pressure, thin-airfoil check, Mach correction, reference-file overlay with RMS error, timing. The PINN Lab shows trained networks (measured on this laptop) next to the FVM reference and the panel method.</li>
        <li><b>What the PINN is and is not:</b> one network is trained per shape, Mach number and angle, so it answers only that condition; for any other condition use the panel-method explorer. In our runs it was more accurate than the FVM on the same mesh, about as accurate as the panel method on smooth cases, and not faster in any like-for-like sense. Neither positivity setting won, and 50×25 runs drift late (see the replay).</li>
        <li><b>Stand-in in the explorer:</b> the explorer's solver is the panel method. Mach uses the Prandtl–Glauert rule, an approximation of the compressible Euler flow NNfoil solves.</li>
        <li><b>Quoted, not measured here:</b> the Mach number of each case (from the paper), the FVM solve times and NeuralFoil's training set size.</li>
        <li><b>Still to come:</b> training a PINN on your drawn shape, ablations against the late drift, a 400×200 PINN and the random airfoils.</li>
      </ul>
      <div className="hint">Scope: inviscid, subsonic flow. A fast screening tool that complements, not replaces, high-fidelity CFD.</div>
    </section>
  );
}
