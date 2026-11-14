// The four control cards on the left: solver, airfoil, flow conditions, reference.
import FvmCredit from './FvmCredit.jsx';

export function SolverCard({ solvers, value, onChange }) {
  const pinn = solvers.find((s) => s.id === 'pinn');
  return (
    <section className="card">
      <h2>1 · Solver</h2>
      <div className="seg" role="group" aria-label="Solver source">
        {solvers.map((s) => (
          // the PINN button is never selectable here: a trained network cannot answer arbitrary inputs (see the note below)
          <button key={s.id} aria-pressed={s.id === value} disabled={!s.available || s.id === 'pinn'} title={s.id === 'pinn' ? 'Trained cases only: see the PINN Lab' : s.available ? undefined : s.note} onClick={() => onChange(s.id)}>
            {s.label}
          </button>
        ))}
      </div>
      {pinn && (
        <div className="hint">
          The NNfoil PINN is not offered here: one network is trained per shape, Mach number and angle, so it answers only that condition and cannot solve the
          inputs below. Its trained cases are in the <a href="#pinn">PINN Lab</a>; for any other condition, use the panel method.
        </div>
      )}
    </section>
  );
}

export function AirfoilCard({ presets, code, custom, message, onPreset, onCustom }) {
  return (
    <section className="card">
      <h2>2 · Airfoil</h2>
      <div className="seg" role="group" aria-label="Airfoil presets">
        {presets.map((p) => (
          <button key={p.code} aria-pressed={p.code === code} onClick={() => onPreset(p.code)}>
            {p.label}
          </button>
        ))}
      </div>
      <div style={{ marginTop: 10 }}>
        <label className="field" htmlFor="custom">Custom 4-digit code</label>
        <input type="text" id="custom" maxLength={4} inputMode="numeric" placeholder="e.g. 2315" aria-describedby="customHint" value={custom} onChange={(e) => onCustom(e.target.value)} />
        <div className="hint err" id="customHint" role="status">{message}</div>
      </div>
      <div className="hint">Final demo lists only shapes that have a trained model (NACA 0012, 2412 and the bundled random airfoils).</div>
    </section>
  );
}

export function FlowCard({ alpha, mach, onAlpha, onMach, idp = '', title = '3 · Flow conditions', hint = 'Subsonic only (Ma < 0.7). Reference data was computed at 4° and Mach 0.4.' }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      <label className="field" htmlFor={`${idp}alpha`}>Angle of attack <span className="val" id={`${idp}alphaV`}>{alpha.toFixed(1)}°</span></label>
      <input type="range" id={`${idp}alpha`} min="-10" max="12" step="0.5" value={alpha} onChange={(e) => onAlpha(+e.target.value)} />
      <label className="field" htmlFor={`${idp}mach`} style={{ marginTop: 14 }}>Mach number <span className="val" id={`${idp}machV`}>{mach.toFixed(2)}</span></label>
      <input type="range" id={`${idp}mach`} min="0" max="0.7" step="0.05" value={mach} onChange={(e) => onMach(+e.target.value)} />
      <div className="hint">{hint}</div>
    </section>
  );
}

export function ReferenceCard({ status, onLoadSample, onClear, onFile, sampleAvailable = true }) {
  return (
    <section className="card">
      <h2>4 · Compare with reference</h2>
      <div className="upload">
        {sampleAvailable && <button className="btn" id="loadSample" onClick={onLoadSample}>Load bundled FVM sample</button>}
        <button className="btn" id="clearRef" onClick={onClear}>Clear</button>
      </div>
      <div style={{ marginTop: 8 }}>
        <label htmlFor="file" className="hint">Or load your own NNfoil <code>*_Cp.dat</code> file:</label>
        <input type="file" id="file" accept=".dat,.txt" onChange={(e) => { onFile(e.target.files[0]); e.target.value = ''; }} />
      </div>
      {sampleAvailable ? <FvmCredit /> : <div className="hint">The bundled FVM sample is not part of this deployment (strict mode: NNfoil's reference data is withheld); your own file works as usual.</div>}
      <div className={`status ${status.kind}`} id="refStatus" role="status">{status.text}</div>
    </section>
  );
}
