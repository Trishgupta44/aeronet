import { useCallback, useEffect, useState } from 'react';
import { getAirfoils, getSampleReference, getSolvers, parseReference } from './api.js';
import { AirfoilCard, FlowCard, ReferenceCard, SolverCard } from './components/ControlCards.jsx';
import CpCard from './components/CpCard.jsx';
import FieldCard from './components/FieldCard.jsx';
import { ComparisonCard, PipelineCard, ResultsCard, TrainingCard } from './components/InfoCards.jsx';
import { useSolve } from './hooks/useSolve.js';
import { checkNacaCode } from './utils/naca.js';

// Shown until (or if) the backend answers, so the layout never jumps.
const FALLBACK_SOLVERS = [
  { id: 'panel', label: 'Panel method', available: true },
  { id: 'pinn', label: 'NNfoil PINN', available: false, note: 'PINN option unlocks when a trained checkpoint is added (plan Day 5).' },
];
const FALLBACK_PRESETS = [
  { code: '0012', label: 'NACA 0012' },
  { code: '2412', label: 'NACA 2412' },
  { code: '4412', label: 'NACA 4412' },
];

const params = new URLSearchParams(window.location.search);
if (params.has('light')) document.documentElement.dataset.theme = 'light'; // ?light forces the light theme (printed figures)

export default function App() {
  const [solvers, setSolvers] = useState(FALLBACK_SOLVERS);
  const [presets, setPresets] = useState(FALLBACK_PRESETS);
  const [solver, setSolver] = useState('panel');
  const [code, setCode] = useState('0012');
  const [custom, setCustom] = useState('');
  const [customMsg, setCustomMsg] = useState('');
  const [alpha, setAlpha] = useState(4);
  const [mach, setMach] = useState(0.4);
  const [reference, setReference] = useState(null);
  const [refStatus, setRefStatus] = useState({ kind: '', text: '' });

  const { result, error } = useSolve({ solver, code, alpha, mach, reference });

  useEffect(() => {
    getSolvers().then(setSolvers).catch(() => {});
    getAirfoils().then(setPresets).catch(() => {});
  }, []);

  const onPreset = (c) => {
    setCustom('');
    setCustomMsg('');
    setCode(c);
  };
  const onCustom = (v) => {
    setCustom(v);
    const check = checkNacaCode(v);
    setCustomMsg(check.message);
    if (check.valid) setCode(v);
  };

  const loadSample = useCallback(async () => {
    try {
      const s = await getSampleReference();
      setCode(s.condition.code);
      setCustom('');
      setCustomMsg('');
      setAlpha(s.condition.alpha);
      setMach(s.condition.mach);
      setReference({ name: s.name, points: s.points });
      setRefStatus({ kind: 'ok', text: `Loaded ${s.count} points from ${s.name}.` });
    } catch (e) {
      setRefStatus({ kind: 'err', text: e.message });
    }
  }, []);

  const onFile = async (file) => {
    if (!file) return;
    try {
      const ref = await parseReference(await file.text(), file.name);
      setReference({ name: ref.name, points: ref.points });
      setRefStatus({ kind: 'ok', text: `Loaded ${ref.count} points from ${ref.name}.` });
    } catch (e) {
      setRefStatus({ kind: 'err', text: e.message });
    }
  };

  const onClear = () => {
    setReference(null);
    setRefStatus({ kind: '', text: '' });
  };

  useEffect(() => {
    if (params.has('sample')) loadSample(); // ?sample opens the page with the reference overlay loaded
  }, [loadSample]);

  return (
    <div className="wrap">
      <header>
        <h1>AeroNet</h1>
        <span className="badge">PROTOTYPE</span>
      </header>
      <p className="sub">Airfoil flow explorer: pick a shape and angle of attack, see the pressure field and lift instantly.</p>
      <div className="notice">
        <b>Read this first.</b> This page shows how the final demo will look and behave. The solver running here (in the Python backend) is a
        classical <b>panel method</b> (real physics, no neural network) standing in for the trained PINN. On Day 5 of the plan, the PINN results
        replace it; the layout, controls and checks stay the same.
      </div>
      {error && (
        <div className="notice error-banner" role="alert">
          {error}
          {error.startsWith('Cannot reach') && (
            <>
              {' '}Start the backend: <code>uvicorn aeronet.main:app --port 8000</code> (from <code>app/backend</code>).
            </>
          )}
        </div>
      )}

      <div className="layout">
        <aside className="stack">
          <SolverCard solvers={solvers} value={solver} onChange={setSolver} />
          <AirfoilCard presets={presets} code={code} custom={custom} message={customMsg} onPreset={onPreset} onCustom={onCustom} />
          <FlowCard alpha={alpha} mach={mach} onAlpha={setAlpha} onMach={setMach} />
          <ReferenceCard status={refStatus} onLoadSample={loadSample} onClear={onClear} onFile={onFile} />
        </aside>
        <main className="stack">
          <FieldCard result={result} />
          <ResultsCard result={result} />
          <CpCard result={result} reference={reference} />
          <ComparisonCard result={result} />
          <TrainingCard />
          <PipelineCard />
        </main>
      </div>
    </div>
  );
}
