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
import BackendBanner from './components/BackendBanner.jsx';
import OptimizeCard from './components/OptimizeCard.jsx';
import ShowdownCard from './components/ShowdownCard.jsx';
import TradeoffCard from './components/TradeoffCard.jsx';
import ViscousCard from './components/ViscousCard.jsx';
import SiteHeader from './components/SiteHeader.jsx';
import { DrawThumb, ExploreThumb, LabThumb, ResultsThumb } from './components/Thumbs.jsx';
import { HEADLINE } from './data/headline.js';
import { useSolve } from './hooks/useSolve.js';
import { checkNacaCode } from './utils/naca.js';
import { NAV_TITLES, pageFromHash } from './utils/route.js';
import { readState, writeState } from './utils/urlState.js';

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
// ?theme=light|dark (or the older ?light) forces a theme for one visit: handled in theme/theme.js
const PRESENT = params.has('present'); // ?present: kiosk mode that cycles through the demo cases
if (PRESENT) document.documentElement.classList.add('present');
const START = readState(window.location.search);

// The cases ?present cycles through, 8 seconds each.
const PRESENT_CYCLE = [
  { tab: 'explore', code: '0012', alpha: 4, mach: 0.4 },
  { tab: 'explore', code: '2412', alpha: 6, mach: 0.3 },
  { tab: 'draw', shape: 'bird', alpha: 4, mach: 0 },
  { tab: 'draw', shape: 'sedan', alpha: 0, mach: 0 },
  { tab: 'draw', shape: 'circle', alpha: 0, mach: 0 },
];
const PRESENT_MS = 8000;

export default function App() {
  const [solvers, setSolvers] = useState(FALLBACK_SOLVERS);
  const [presets, setPresets] = useState(FALLBACK_PRESETS);
  const [solver, setSolver] = useState('panel');
  const [code, setCode] = useState(START.code);
  const [custom, setCustom] = useState(FALLBACK_PRESETS.some((p) => p.code === START.code) ? '' : START.code);
  const [customMsg, setCustomMsg] = useState('');
  const [alpha, setAlpha] = useState((START.tab === 'explore' ? START.alpha : undefined) ?? 4);
  const [mach, setMach] = useState((START.tab === 'explore' ? START.mach : undefined) ?? 0.4);
  const [reference, setReference] = useState(null);
  const [refStatus, setRefStatus] = useState({ kind: '', text: '' });
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [drawLayers, setDrawLayers] = useState(DEFAULT_LAYERS);
  const [paused, setPaused] = useState(prefersReducedMotion);
  // The page (tab) on show: home, explore, draw, pinn or results. The #hash of the address says which; links are plain #hash links.
  const [page, setPage] = useState(() => (PRESENT ? START.tab : pageFromHash(window.location.hash) ?? (params.has('tab') ? START.tab : 'home')));
  const [tab, setTab] = useState(START.tab); // the section whose state the page address follows (explore or draw)
  const [drawState, setDrawState] = useState(null);
  const [drawRequest, setDrawRequest] = useState(null);

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

  // follow the #hash: a link to a page shows that page from its top; other hashes (#credit) are ordinary anchors
  useEffect(() => {
    const onHash = () => {
      const p = pageFromHash(window.location.hash);
      if (!p) return;
      setPage(p);
      window.scrollTo(0, 0);
    };
    // popstate too: after replaceState has rewritten the query string, Back to an older entry is not always reported as a hashchange
    window.addEventListener('hashchange', onHash);
    window.addEventListener('popstate', onHash);
    return () => {
      window.removeEventListener('hashchange', onHash);
      window.removeEventListener('popstate', onHash);
    };
  }, []);
  useEffect(() => {
    if (page === 'explore' || page === 'draw') setTab(page);
    document.title = page === 'home' ? 'AeroNet: airflow explorer' : `${NAV_TITLES[page]} · AeroNet`;
  }, [page]);

  // the page address follows the section the reader last used (freehand outlines are not stored); only the tool pages carry state
  useEffect(() => {
    if (page !== 'explore' && page !== 'draw') return;
    const s =
      tab === 'draw' && drawState
        ? { tab, code, alpha: drawState.alpha, mach: drawState.mach, shape: drawState.shape, lifting: drawState.lifting, ground: drawState.ground }
        : { tab: 'explore', code, alpha, mach, shape: '', lifting: undefined, ground: '' };
    const { pathname, search, hash } = window.location;
    const next = pathname + writeState(search, s) + hash;
    if (next !== pathname + search + hash) window.history.replaceState(null, '', next);
  }, [page, tab, code, alpha, mach, drawState]);

  // ?present: cycle through the demo cases until the first user input
  useEffect(() => {
    if (!PRESENT) return undefined;
    let stopped = false;
    const stop = () => {
      stopped = true;
    };
    const events = ['pointerdown', 'keydown', 'wheel', 'touchstart'];
    events.forEach((e) => window.addEventListener(e, stop, { once: true, passive: true }));
    const apply = (i) => {
      const c = PRESENT_CYCLE[i];
      setTab(c.tab);
      setPage(c.tab);
      if (c.tab === 'explore') {
        setCustom('');
        setCode(c.code);
        setAlpha(c.alpha);
        setMach(c.mach);
      } else {
        setDrawRequest({ id: c.shape, alpha: c.alpha, mach: c.mach, nonce: i + Math.random() });
      }
    };
    apply(0);
    let i = 0;
    const id = setInterval(() => {
      if (stopped) {
        clearInterval(id);
        return;
      }
      i = (i + 1) % PRESENT_CYCLE.length;
      apply(i);
    }, PRESENT_MS);
    return () => {
      clearInterval(id);
      events.forEach((e) => window.removeEventListener(e, stop));
    };
  }, []);

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
      <BackendBanner />
      <SiteHeader page={page} />
      {page === 'home' && (
        <Hero caption={result ? `Live: panel-method flow around NACA ${result.code} at ${result.alpha}°, the case selected in Explore` : undefined}>
          {result && (
            <div className="hero-flow">
              <FlowStage result={result} layers={HERO_LAYERS} paused={paused} hero />
            </div>
          )}
        </Hero>
      )}
      <main>
        <Section
          id="about"
          cls="hide-present"
          hidden={page !== 'home'}
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
              <p className="l">solve time of the panel method for the case selected in Explore, on this machine</p>
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

        <Section id="explore" eyebrow="Airfoil flow explorer" title="Explore" hidden={page !== 'explore'}>
          <details className="real hide-present">
            <summary>What is real here</summary>
            <ul>
              <li><b>Real now:</b> the airfoil geometry, the panel solver (Hess–Smith, Python backend), the pressure field and curve, Cl from surface pressure, the thin-airfoil check (NACA codes only), the Mach correction and the solve time, which is measured on every request.</li>
              <li><b>Stand-in:</b> the solver is a classical panel method (real physics, no neural network). The trained PINN replaces it when its results are exported; the layout and controls stay the same.</li>
              <li><b>Idealized:</b> inviscid, subsonic (Prandtl–Glauert) and 2D. There is no separation or wake, so the drag shown is a numerical error, not a prediction, and cars show no drag value at all.</li>
              <li><b>Quoted, not measured here:</b> the NeuralFoil timings in the comparison table come from its paper.</li>
            </ul>
          </details>
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
          <div className="layout" onPointerDownCapture={() => setTab('explore')} onKeyDownCapture={() => setTab('explore')}>
            <aside className="stack">
              <SolverCard solvers={solvers} value={solver} onChange={setSolver} />
              <AirfoilCard presets={presets} code={code} custom={custom} message={customMsg} onPreset={onPreset} onCustom={onCustom} />
              <FlowCard alpha={alpha} mach={mach} onAlpha={setAlpha} onMach={setMach} />
              <ReferenceCard status={refStatus} onLoadSample={loadSample} onClear={onClear} onFile={onFile} />
            </aside>
            <div className="stack">
              <CaseView caseName={`NACA ${code}`} polarSource={{ code, mach }} result={result} reference={reference} layers={layers} onLayersChange={setLayers} paused={paused} onPausedChange={setPaused} />
              <OptimizeCard code={code} />
              <ViscousCard code={code} alpha={alpha} /> {/* NACA airfoils only: hide this for cars, bluff bodies and drawn shapes */}
            </div>
          </div>
        </Section>

        <Section id="draw" band eyebrow="Any closed shape" title="Draw" hidden={page !== 'draw'}>
          <DrawSection layers={drawLayers} onLayersChange={setDrawLayers} paused={paused} onPausedChange={setPaused}
            initial={START.tab === 'draw' ? START : { shape: START.shape, lifting: START.lifting, ground: START.ground }}
            request={drawRequest} onState={setDrawState} onActivate={() => setTab('draw')} />
        </Section>

        <Section id="pinn" cls="hide-present" hidden={page !== 'pinn'} eyebrow="Neural solver" title="PINN Lab">
          <div className="results-stack">
            <ShowdownCard />
            <TrainingCard />
          </div>
        </Section>

        <Section id="overview" cls="hide-present" hidden={page !== 'home'} band eyebrow="Four ways in" title="What you can do">
          <CardGrid items={cards} />
        </Section>

        <Section id="results" cls="hide-present" hidden={page !== 'results'} eyebrow="Evidence" title="Results">
          <div className="results-stack">
            <TradeoffCard />
            <ComparisonCard result={result} />
            <PipelineCard />
          </div>
        </Section>
      </main>
      <SiteFooter />
    </>
  );
}
