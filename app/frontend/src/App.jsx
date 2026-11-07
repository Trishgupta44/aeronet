import { useCallback, useEffect, useState } from 'react';
import { getAirfoils, getSampleReference, getSolvers, parseReference } from './api.js';
import { AirfoilCard, FlowCard, ReferenceCard, SolverCard } from './components/ControlCards.jsx';
import CaseView from './components/CaseView.jsx';
import DrawSection from './components/DrawSection.jsx';
import { DEFAULT_LAYERS, FlowStage, HERO_LAYERS, prefersReducedMotion } from './components/FlowCanvas.jsx';
import { ComparisonCard, PipelineCard, TrainingCard } from './components/InfoCards.jsx';
import CardGrid from './components/CardGrid.jsx';
import Hero from './components/Hero.jsx';
import Section from './components/Section.jsx';
import SiteFooter from './components/SiteFooter.jsx';
import SiteHeader from './components/SiteHeader.jsx';
import { DrawThumb, ExploreThumb, LabThumb, ResultsThumb } from './components/Thumbs.jsx';
import { HEADLINE } from './data/headline.js';
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
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [drawLayers, setDrawLayers] = useState(DEFAULT_LAYERS);
  const [paused, setPaused] = useState(prefersReducedMotion);

  const { result, error } = useSolve({ kind: 'naca', solver, code, alpha, mach, reference });

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

  const cards = [
    { title: 'Explore', href: '#explore', thumb: <ExploreThumb />, text: 'Pick an airfoil, set the angle and Mach number, and watch the pressure field and lift update.' },
    { title: 'Draw', href: '#draw', thumb: <DrawThumb />, text: 'Sketch any closed 2D shape with a mouse, pen or finger, or pick a car, a bird wing or a UIUC airfoil file, and get its flow.' },
    { title: 'PINN Lab', href: '#pinn', thumb: <LabThumb />, text: 'Trained neural solutions next to the finite-volume reference. They appear here once the runs are exported.' },
    { title: 'Results', href: '#results', thumb: <ResultsThumb />, text: 'What is measured, what is quoted from papers and what is still to be done, side by side.' },
  ];

  return (
    <>
      <SiteHeader />
      <Hero caption={result ? `Live: panel-method flow around NACA ${result.code} at ${result.alpha}°, the case selected below` : undefined}>
        {result && (
          <div className="hero-flow">
            <FlowStage result={result} layers={HERO_LAYERS} paused={paused} hero />
          </div>
        )}
      </Hero>
      <main>
        <Section
          id="about"
          band
          eyebrow="An airfoil flow explorer"
          title="We are AeroNet"
          aside={
            <>
              <p>
                AeroNet shows how air moves around a wing section. Pick a shape and an angle, and see the pressure field, the lift and how a
                neural solver compares with classical methods. It is a screening tool: inviscid, subsonic and honest about what it does not do.
              </p>
              <a className="button button--primary" href="#results">About the project</a>
            </>
          }
        >
          <div className="about-numbers">
            <div className="bignum">
              <div className="n">{HEADLINE.pinnError.value}</div>
              <p className="l">{HEADLINE.pinnError.label}</p>
              <span className="tag tag--measured">{HEADLINE.pinnError.tag}</span>
              <span className="src">{HEADLINE.pinnError.source}</span>
            </div>
            <div className="bignum">
              <div className="n">{result ? `${result.solve_ms.toFixed(1)} ms` : '–'}</div>
              <p className="l">solve time of the panel method for the case shown below, on this machine</p>
              <span className="tag tag--measured">measured now</span>
              <span className="src">timed inside the backend for every request</span>
            </div>
            <div className="bignum">
              <div className="n">{HEADLINE.mesh.value}</div>
              <p className="l">{HEADLINE.mesh.label}</p>
              <span className="tag tag--measured">{HEADLINE.mesh.tag}</span>
              <span className="src">{HEADLINE.mesh.source}</span>
            </div>
          </div>
        </Section>

        <Section id="explore" eyebrow="Airfoil flow explorer" title="Explore">
          <div className="notice">
            <b>Read this first.</b> This page shows how the final demo will look and behave. The solver running here (in the Python backend) is a
            classical <b>panel method</b> (real physics, no neural network) standing in for the trained PINN. When the PINN results are exported,
            they replace it; the layout, controls and checks stay the same.
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
            <div className="stack">
              <CaseView result={result} reference={reference} layers={layers} onLayersChange={setLayers} paused={paused} onPausedChange={setPaused} />
            </div>
          </div>
        </Section>

        <Section id="draw" band eyebrow="Any closed shape" title="Draw">
          <DrawSection layers={drawLayers} onLayersChange={setDrawLayers} paused={paused} onPausedChange={setPaused} />
        </Section>

        <Section id="pinn" eyebrow="Neural solver" title="PINN Lab">
          <div className="results-stack">
            <div className="placeholder">Trained cases appear here once the PINN runs are exported. Nothing is shown now on purpose: there is no fake data on this page.</div>
            <TrainingCard />
          </div>
        </Section>

        <Section id="overview" band eyebrow="Four ways in" title="What you can do">
          <CardGrid items={cards} />
        </Section>

        <Section id="results" eyebrow="Evidence" title="Results">
          <div className="results-stack">
            <ComparisonCard result={result} />
            <PipelineCard />
          </div>
        </Section>
      </main>
      <SiteFooter />
    </>
  );
}
