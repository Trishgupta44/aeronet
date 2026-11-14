// Results card, PINN training history and the folded scope note of the Results page.
import { fmt } from '../utils/format.js';
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

// The honest scope note of the Results page, folded away: what is measured here, what is a stand-in, what is quoted and what is still to do.
export function RealFold() {
  const rows = [
    ['Real now', 'airfoil geometry, panel solver (Python backend), Cp field and curve, Cl from pressure, thin-airfoil check, Mach correction, reference-file overlay with RMS error, timing. The PINN Lab shows trained networks (measured on this laptop) next to the FVM reference and the panel method.'],
    ['What the PINN is and is not', 'one network is trained per shape, Mach number and angle, so it answers only that condition; for any other condition use the panel-method explorer. In our runs it was more accurate than the FVM on the same mesh, about as accurate as the panel method on smooth cases, and not faster in any like-for-like sense. Neither positivity setting won, and 50×25 runs drift late.'],
    ['Stand-in in the explorer', 'the explorer’s solver is the panel method. Mach uses the Prandtl–Glauert rule, an approximation of the compressible Euler flow NNfoil solves.'],
    ['Quoted, not measured here', 'the Mach number of each case (from the paper), the FVM solve times and NeuralFoil’s training set size.'],
    ['Still to come', 'training a PINN on your drawn shape, ablations against the late drift, a 400×200 PINN and the random airfoils.'],
  ];
  return (
    <details className="fold" id="what-is-real">
      <summary><span>What is real, what is a stand-in</span><span className="fold-hint">scope and honesty notes</span></summary>
      <div className="fold-body">
        <ol className="steps">
          <li className="done"><b>Demo ✓</b>layout, controls, Cp field, Cl checks, draw-any-shape</li>
          <li className="done"><b>Verify ✓</b>NNfoil code reviewed and fixed</li>
          <li className="done"><b>Train ✓</b>NNfoil on this laptop’s CPU: 14 exported runs, 50×25 to 200×100 meshes</li>
          <li className="done"><b>Validate ✓</b>PINN against FVM error tables, speed table, stop-rule test</li>
        </ol>
        <dl className="real-grid">
          {rows.map(([k, v]) => (
            <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
          ))}
        </dl>
        <p className="hint">Scope: inviscid, subsonic flow. A fast screening tool that complements, not replaces, high-fidelity CFD.</p>
      </div>
    </details>
  );
}
