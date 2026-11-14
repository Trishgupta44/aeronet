import { useEffect, useMemo, useRef, useState } from 'react';
import { getPinnCase, getPinnCases } from '../api.js';
import { fmt, fmtMs, fmtSec, meshLabel } from '../utils/format.js';
import { caseGroups, caseToResult, fvmView, replayCurve, roundRms, shortVariant } from '../utils/pinnCase.js';
import FlowCanvas, { DEFAULT_LAYERS } from './FlowCanvas.jsx';
import FvmCredit from './FvmCredit.jsx';
import FvmWithheldNote from './FvmWithheldNote.jsx';
import { TrainingCard } from './InfoCards.jsx';
import PinnCpChart from './PinnCpChart.jsx';
import Tag from './Tag.jsx';

const DEFAULT_CASE = 'naca0012_100x50_a4_m04_gs'; // the headline case: grid-sequenced 100x50, 4 deg, Mach 0.4
const STEP_MS = 600; // replay speed: one training round per step
const CACHE_MAX = 4; // parsed cases kept in memory (each holds about 1 MB of field numbers)

const startParams = new URLSearchParams(typeof window !== 'undefined' ? window.location.search : '');

function CasePicker({ list, value, onPick }) {
  const groups = useMemo(() => caseGroups(list), [list]);
  return (
    <section className="card">
      <h2>Pick a trained case</h2>
      <p className="hint" style={{ marginTop: 0 }}>
        Each case is one network trained by us for one shape, Mach number and angle (NNfoil, after Cao, Song and Zhang). Results are{' '}
        <Tag kind="measured" /> on this laptop; the Mach number of each case is <Tag kind="quoted" /> from the paper, because the Cp files do not store it.
      </p>
      {groups.map((g) => (
        <div className="case-group" key={g.shape}>
          <h3 className="case-heading">{g.heading}</h3>
          <div className="case-grid" role="group" aria-label={`${g.heading} cases`}>
            {g.cases.map((c) => {
              const variant = shortVariant(c.label);
              return (
                <button key={c.id} type="button" className="case" aria-pressed={c.id === value} onClick={() => onPick(c.id)}>
                  <span className="case-main">{meshLabel(c.mesh)}{variant ? ` · ${variant}` : ''}</span>
                  <span className="case-sub mono">Ma {c.mach} · {c.alpha}° · RMS {fmt(c.rms, c.rms >= 1 ? 2 : 4)}</span>
                  <span className="case-tags">
                    <Tag kind={c.source?.results ?? 'measured'} />
                    {c.source?.condition === 'quoted' && <Tag kind="quoted">Ma quoted</Tag>}
                    {c.collapsed && <span className="tag tag--collapsed">collapsed</span>}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </section>
  );
}

function Tile({ k, v, n, tag = 'measured', title }) {
  return (
    <div className="metric" title={title}>
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      <div className="n">{n}</div>
      {tag && <Tag kind={tag} />}
    </div>
  );
}

function Tiles({ c }) {
  const m = c.metrics ?? {};
  const t = c.timing ?? {};
  const { ref, same } = fvmView(c); // the names and numbers are there in a public deployment too, only the FVM curves are not
  const panel = c.panel;
  return (
    <div className="metrics metrics--lab" aria-label="Numbers for this case">
      <Tile k="Cp RMS error" v={ref ? fmt(m.rms, m.rms >= 1 ? 2 : 4) : '–'}
        n={ref ? `vs ${ref.name} · max ${fmt(m.max, 3)} · ${m.n} points` : 'no reference solution for this case'} />
      <Tile k="Lift coefficient Cl" v={fmt(m.cl, 3)} n={`reference ${fmt(m.cl_ref, 3)}${ref ? ` (${ref.name})` : ''}${panel ? ` · panel ${fmt(panel.cl, 3)}` : ''}`} />
      <Tile k="Drag Cd (error only)" v={fmt(m.cd, 4)} n="inviscid flow: the ideal value is 0" />
      <Tile k="FVM, same mesh" v={same ? fmt(same.rms, 4) : '–'}
        n={same ? `${same.name} · Cp RMS vs 800×400 · Cl ${fmt(same.cl, 3)}` : 'no same-mesh FVM run for this case'} tag={same ? 'measured' : null} />
      <Tile k="Panel method" v={panel?.metrics ? fmt(panel.metrics.rms, 4) : '–'}
        n={panel ? `Cp RMS · Cl ${fmt(panel.cl, 3)} · ${panel.name}` : 'no panel solution'} tag={panel ? 'measured' : null} />
      <Tile k="Training time" v={t.train_s != null ? fmtSec(t.train_s) : 'withheld'}
        n={t.train_s != null ? `${c.log?.round?.length ?? '?'} rounds logged · ${t.threads} threads${t.note ? ` · ${t.note}` : ''}` : (t.note ?? 'no time recorded')}
        tag={t.train_s != null ? 'measured' : null} title={t.cpu} />
      <Tile k="Inference time" v={t.infer_ms != null ? fmtMs(t.infer_ms) : 'not timed'}
        n={t.infer_ms != null ? 'one forward pass over every mesh node · mean of 100 calls' : 'only the NACA 0012 runs were timed (experiment E4)'}
        tag={t.infer_ms != null ? 'measured' : null} title={t.infer_note} />
    </div>
  );
}

function CollapsedBanner({ c }) {
  const cps = c.surface.cp;
  const lo = Math.min(...cps);
  const hi = Math.max(...cps);
  return (
    <div className="notice error-banner" role="note">
      <b>This run collapsed to the trivial solution: shown on purpose.</b> {c.metrics.collapsed_reason ? `${c.metrics.collapsed_reason}. ` : ''}
      {lo === hi ? <>The network's wall Cp is the same number ({lo.toFixed(2)}) at every point, so its Cl is {fmt(c.metrics.cl, 3)}. </> : ''}
      The loss in the training history is astronomically large for the same reason. Other seeds and settings of this case did train (see the other NACA 2412 cases).
    </div>
  );
}

function Replay({ c, idx, setIdx, playing, setPlaying, stats }) {
  const snaps = c.snapshots ?? [];
  const last = snaps.length - 1;
  const cur = replayCurve(c, idx);
  if (last < 1 || !cur) return <p className="hint">This case has no per-round curves to replay.</p>;
  const total = snaps[last].round;
  const rms = stats[idx];
  const known = stats.map((v, i) => [v, i]).filter(([v]) => v != null);
  const r3 = snaps.findIndex((s) => s.round === 3);
  const best = known.length ? known.reduce((a, b) => (b[0] < a[0] ? b : a)) : null;
  const onPlay = () => {
    if (playing) return setPlaying(false);
    if (idx >= last) setIdx(0);
    return setPlaying(true);
  };
  return (
    <div className="replay">
      <div className="replay-bar">
        <button type="button" className="btn replay-btn" onClick={onPlay} aria-label={playing ? 'Pause the replay' : 'Play the training replay'}>
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
        <input type="range" min="0" max={last} step="1" value={idx} aria-label="Training round" aria-valuetext={`round ${cur.round} of ${total}`}
          onChange={(e) => { setPlaying(false); setIdx(+e.target.value); }} />
      </div>
      <p className="replay-read mono" id="replayRead">
        <b>Round {cur.round} of {total}</b>
        {cur.cl != null && <> · Cl {fmt(cur.cl, 3)} <span className="hint">(reference {fmt(c.metrics?.cl_ref, 3)})</span></>}
        {rms != null && <> · Cp RMS {fmt(rms, rms >= 1 ? 2 : 4)}</>}
        {idx === last && <> · final round</>}
      </p>
      {best && (
        <p className="hint">
          <Tag kind="computed" /> Cp RMS by round, from each round's wall Cp with the exported definition:
          {r3 >= 0 && stats[r3] != null && <> round 3 {fmt(stats[r3], 4)},</>} final round {fmt(stats[last], 4)}, lowest {fmt(best[0], 4)} at round {snaps[best[1]].round}.
          The lowest round is found with the reference answer, so no real training run could pick it: the practice is a fixed early stop (rounds 3 to 5 in our runs).
        </p>
      )}
    </div>
  );
}

export default function PinnLab({ paused, onPausedChange }) {
  const [list, setList] = useState(null);
  const [listError, setListError] = useState('');
  const [selId, setSelId] = useState(null);
  const [c, setC] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [show, setShow] = useState({ ref: true, same: false, panel: true });
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const cache = useRef(new Map());
  const firstLoad = useRef(true);

  useEffect(() => {
    getPinnCases()
      .then((l) => {
        setList(l);
        const want = startParams.get('pinn');
        setSelId((l.find((x) => x.id === want) ?? l.find((x) => x.id === DEFAULT_CASE) ?? l[0])?.id ?? null);
      })
      .catch((e) => setListError(e.message));
  }, []);

  useEffect(() => {
    if (!selId) return undefined;
    let live = true;
    setLoadError('');
    const done = (data) => {
      if (!live) return;
      setC(data);
      setPlaying(false);
      const last = (data.snapshots?.length ?? 1) - 1;
      const want = firstLoad.current ? Number(startParams.get('round')) : NaN;
      firstLoad.current = false;
      setIdx(Number.isFinite(want) && want > 0 ? Math.max(0, Math.min(last, want - 1)) : Math.max(0, last)); // start on the final round, which is what the numbers describe
    };
    const hit = cache.current.get(selId);
    if (hit) {
      done(hit);
    } else {
      getPinnCase(selId)
        .then((data) => {
          cache.current.set(selId, data);
          if (cache.current.size > CACHE_MAX) cache.current.delete(cache.current.keys().next().value);
          done(data);
        })
        .catch((e) => live && setLoadError(e.message));
    }
    return () => {
      live = false;
    };
  }, [selId]);

  const last = (c?.snapshots?.length ?? 1) - 1;
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => setIdx((i) => Math.min(last, i + 1)), STEP_MS);
    return () => clearInterval(id);
  }, [playing, last]);
  useEffect(() => {
    if (playing && idx >= last) setPlaying(false);
  }, [playing, idx, last]);

  const result = useMemo(() => (c ? caseToResult(c) : null), [c]);
  const stats = useMemo(() => (c ? roundRms(c) : []), [c]);
  const fvm = useMemo(() => (c ? fvmView(c) : null), [c]);
  const rmsSeries = useMemo(() => (c?.snapshots ? { rounds: c.snapshots.map((s) => s.round), values: stats } : null), [c, stats]);
  const cur = c ? replayCurve(c, idx) : null;
  const cp = cur?.cp ?? c?.surface?.cp ?? [];
  const stale = c && selId && c.id !== selId && !loadError;

  const toggle = (key) => setShow((s) => ({ ...s, [key]: !s[key] }));

  if (listError || (list && list.length === 0)) {
    return (
      <section className="card">
        <h2>PINN Lab</h2>
        <div className="placeholder">
          Trained cases are not available from this server{listError ? ` (${listError})` : ''}. Nothing is shown on purpose: there is no fake data on this page.
        </div>
      </section>
    );
  }
  if (!list) return <section className="card"><h2>PINN Lab</h2><div className="status" role="status">Loading trained cases…</div></section>;

  return (
    <div className="lab">
      <div className="notice" role="note">
        <b>A trained network answers only its trained condition.</b> One network is trained per shape, Mach number and angle, so you cannot ask it for another
        condition. For any other condition use the <a href="#explore">panel-method explorer</a>. Every number below is <Tag kind="measured" /> on one laptop unless it is
        tagged <Tag kind="quoted" /> or <Tag kind="computed" />.
      </div>
      <CasePicker list={list} value={selId} onPick={setSelId} />
      {loadError && <div className="notice error-banner" role="alert">{loadError}</div>}
      {c && (
        <div className={stale ? 'lab-case dim' : 'lab-case'} aria-busy={!!stale}>
          <p className="lab-title">
            <b>{c.label}</b>
            <span className="hint"> · {c.source?.condition === 'quoted' ? 'condition quoted: ' : 'condition: '}{c.source?.condition_note ?? `Mach ${c.mach}, ${c.alpha}°`}</span>
          </p>
          {c.metrics?.collapsed && <CollapsedBanner c={c} />}
          <Tiles c={c} />
          <div className="lab-grid">
            <FlowCanvas idp="pinn-" result={result} layers={layers} onLayersChange={setLayers} paused={paused} onPausedChange={onPausedChange}
              title="PINN pressure field (Cp)"
              hint="The colours, isobars and particles all come from the trained network's own pressure and velocity (the final round). Inviscid Euler flow, no separation or wake."
              footer={<p className="src">Network output on the mesh, resampled onto the app's regular grid. Hover the picture to read Cp and speed.{c.field_note ? ` ${c.field_note}` : ''}</p>} />
            <section className="card">
              <h2>{fvm.ref?.points || fvm.same?.points ? 'Wall pressure: PINN against FVM and panel method' : 'Wall pressure: PINN against the panel method'}</h2>
              <div className="seg" role="group" aria-label="Curves to compare">
                {fvm.ref?.points && <button type="button" aria-pressed={show.ref} onClick={() => toggle('ref')}>{fvm.ref.name}</button>}
                {fvm.same?.points && <button type="button" aria-pressed={show.same} onClick={() => toggle('same')}>{fvm.same.name}, same mesh</button>}
                {c.panel && <button type="button" aria-pressed={show.panel} onClick={() => toggle('panel')}>Panel method</button>}
              </div>
              <PinnCpChart c={c} cp={cp} show={show} />
              {fvm.withheld ? <FvmWithheldNote /> : (fvm.ref?.points || fvm.same?.points) && <FvmCredit />}
              <div className="legend">
                <span><i style={{ background: 'var(--up)' }} />PINN upper surface</span>
                <span><i style={{ background: 'var(--lo)' }} />PINN lower surface</span>
                {show.ref && fvm.ref?.points && <span><i style={{ background: 'transparent', border: '1.5px solid var(--ref)' }} />{fvm.ref.name} (reference)</span>}
                {show.same && fvm.same?.points && <span><i style={{ background: 'var(--ok)', borderRadius: 0 }} />{fvm.same.name}</span>}
                {show.panel && c.panel && <span><i style={{ background: 'transparent', border: '1.5px dashed var(--muted)' }} />panel method</span>}
              </div>
              <h2 className="replay-title">Training replay: watch it learn</h2>
              <Replay c={c} idx={idx} setIdx={setIdx} playing={playing} setPlaying={setPlaying} stats={stats} />
            </section>
          </div>
          <TrainingCard log={c.log} clRef={c.metrics?.cl_ref ?? null} rms={rmsSeries} round={cur?.round ?? null} />
          <p className="src">
            Source: app/backend/data/pinn_cases/{c.id}.json, exported from results/runs/{c.id}. {c.config?.positivity ? 'Positivity on' : 'Positivity off'}, seed {c.config?.seed ?? '?'},
            {' '}{c.log?.round?.length ?? '?'} rounds logged{c.config?.failed_rounds ? `, ${c.config.failed_rounds} failed (the optimiser restarted)` : ''}. The Cp error is the root mean
            square over the reference points with 0.02 ≤ x/c ≤ 0.98, each compared with the nearest PINN wall point.
          </p>
        </div>
      )}
    </div>
  );
}
