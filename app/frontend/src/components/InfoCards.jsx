// Results, speed/accuracy table, training-history placeholder and the project pipeline card.

export function ResultsCard({ result, idp = '' }) {
  const dash = '–';
  const custom = result?.code === 'custom';
  const noDrag = result?.ground_y != null; // a car on a road: inviscid flow has no drag, so no number is shown
  return (
    <section className="card">
      <h2>Results</h2>
      <div className="metrics">
        <div className="metric"><div className="k">Lift coefficient Cl</div><div className="v" id={`${idp}mCl`}>{result ? result.cl.toFixed(3) : dash}</div><div className="n">from surface pressure</div></div>
        <div className="metric">
          <div className="k">Thin-airfoil theory</div>
          <div className="v" id={`${idp}mThin`}>{result && result.cl_thin != null ? result.cl_thin.toFixed(3) : custom ? 'n/a' : dash}</div>
          <div className="n">{custom ? 'not an airfoil code' : 'quick sanity check'}</div>
        </div>
        <div className="metric">
          <div className="k">Drag Cd {noDrag ? '' : '(error only)'}</div>
          <div className="v" id={`${idp}mCd`}>{result ? (noDrag ? 'not shown' : result.cd.toFixed(4)) : dash}</div>
          <div className="n">{noDrag ? 'inviscid flow has no drag' : 'ideal value is 0'}</div>
        </div>
        <div className="metric"><div className="k">Solve time</div><div className="v" id={`${idp}mMs`}>{result ? `${result.solve_ms.toFixed(1)} ms` : dash}</div><div className="n">measured on the server</div></div>
      </div>
    </section>
  );
}

export function ComparisonCard({ result }) {
  return (
    <section className="card">
      <h2>Speed and accuracy comparison</h2>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Method</th><th>Setup</th><th>Per new query</th><th>Physics</th><th>Data needed</th></tr></thead>
          <tbody>
            <tr><td className="live">Panel method (this app)</td><td>none</td><td className="live" id="tMs">{result ? `${result.solve_ms.toFixed(1)} ms (measured now)` : '–'}</td><td>potential flow + Prandtl–Glauert</td><td>none</td></tr>
            <tr><td>NNfoil PINN</td><td className="tbd">training time: measure Day 2-4</td><td className="tbd">inference: measure Day 4</td><td>Euler equations</td><td>none</td></tr>
            <tr><td>FVM (bundled Cp files)</td><td>none</td><td className="tbd">full re-solve; quote time from NNfoil paper</td><td>Euler equations</td><td>none</td></tr>
            <tr><td>NeuralFoil <span className="tbd">(paper, Table 3)</span></td><td>pre-trained</td><td>1.2 ms per run (XFoil: 73 ms)</td><td>viscous, learned</td><td>7.9 million XFoil runs</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function TrainingCard() {
  return (
    <section className="card">
      <h2>Training history</h2>
      <div className="placeholder">
        Loss-vs-time curve from the PINN training log appears here once a run is logged (plan Day 2-3).<br />
        Nothing is shown now on purpose: there is no fake data on this page.
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
        <li><b>Verify</b>code review, 3 known issues</li>
        <li><b>Train</b>NNfoil on CPU (small meshes)</li>
        <li><b>Validate</b>PINN vs FVM error tables</li>
      </ol>
      <ul className="plain">
        <li><b>Real now:</b> airfoil geometry, panel solver (Python backend), Cp field and curve, Cl from pressure, thin-airfoil check, Mach correction, reference-file overlay with RMS error, timing.</li>
        <li><b>Stand-in:</b> the solver itself (PINN replaces it). Mach uses the Prandtl–Glauert rule, an approximation of the compressible Euler flow NNfoil solves.</li>
        <li><b>Built since:</b> any closed shape (draw it, pick a preset or import a UIUC <code>.dat</code> file in the Draw section), a road under the shape, the animated flow view.</li>
        <li><b>Still to come (Plan C):</b> the PINN lab with trained cases next to the finite-volume reference, and the PINN solver behind the same controls.</li>
      </ul>
      <div className="hint">Scope: inviscid, subsonic flow. A fast screening tool that complements, not replaces, high-fidelity CFD.</div>
    </section>
  );
}
