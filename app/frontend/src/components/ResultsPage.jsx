import { useCallback, useMemo, useState } from 'react';
import { useCapabilities } from '../hooks/useCapabilities.js';
import { useRoundsE1 } from '../hooks/useResults.js';
import {
  cylinderBars, failureItems, headlineTiles, meshStudy, meshSummary, trajectories, trajectorySummary,
} from '../utils/evidence.js';
import { fmt, meshLabel } from '../utils/format.js';
import BarDotChart from './BarDotChart.jsx';
import FvmCredit from './FvmCredit.jsx';
import FvmWithheldNote from './FvmWithheldNote.jsx';
import LineChart from './LineChart.jsx';
import ResultsTable from './ResultsTable.jsx';
import SpeedCard from './SpeedCard.jsx';
import Tag from './Tag.jsx';
import TradeoffCard from './TradeoffCard.jsx';
import { RealFold } from './InfoCards.jsx';

const f4 = (v) => fmt(v, 4);

function StatBand({ rows }) {
  const tiles = useMemo(() => headlineTiles(rows), [rows]);
  return (
    <ul className="stat-band" aria-label="Headline numbers">
      {tiles.map((t) => (
        <li key={t.id} className={`stat${t.ok ? '' : ' stat--na'}`}>
          <p className="stat-n">{t.value}</p>
          <p className="stat-l">{t.label}</p>
          <p className="stat-c">{t.compare}</p>
          {t.note && <p className="stat-note">{t.note}</p>}
          <div className="stat-f"><Tag kind={t.tag} /><span className="src">{t.source}</span></div>
        </li>
      ))}
    </ul>
  );
}

function Finding({ id, eyebrow, title, tag, source, children, chart }) {
  return (
    <article className="finding card card--evidence" id={id}>
      <div className="finding-copy">
        <p className="eyebrow">{eyebrow}</p>
        <h3 className="card-title">{title}</h3>
        {children}
        <div className="finding-meta"><Tag kind={tag} /><span className="src">{source}</span></div>
      </div>
      <div className="finding-chart">{chart}</div>
    </article>
  );
}

function MeshFinding({ rows }) {
  const study = useMemo(() => meshStudy(rows), [rows]);
  const sum = useMemo(() => meshSummary(study), [study]);
  const groups = useMemo(() => study.groups.map((g) => ({
    id: g.mesh,
    label: meshLabel(g.mesh),
    sub: g.pinn.length ? null : 'no PINN run',
    bar: { value: g.fvm, kind: 'fvm', detail: `FVM ${meshLabel(g.mesh)}: Cp RMS ${f4(g.fvm)} against FVM 800×400 (measured)` },
    dots: g.pinn.map((p) => ({ value: p.rms, positivity: p.positivity, gridSeq: p.gridSeq, detail: `PINN ${meshLabel(g.mesh)}, ${p.label}: Cp RMS ${f4(p.rms)}, final round (measured)` })),
  })), [study]);
  const level = useMemo(() => (study.panel != null ? { value: study.panel, label: `panel method ${f4(study.panel)}`, short: `panel ${f4(study.panel)}` } : null), [study.panel]);
  if (!groups.length) return <MissingFinding eyebrow="E2 · Mesh study" what="The mesh study rows" />;
  return (
    <Finding
      id="finding-mesh" eyebrow="E2 · Mesh study"
      title={sum.allLower ? 'On the same mesh, the PINN is more accurate than the FVM' : 'The PINN against the FVM on the same mesh'}
      tag="measured" source="summary.csv, E2"
      chart={(
        <BarDotChart
          title="Cp RMS against FVM 800×400, by mesh (log scale)" groups={groups} log yMin={0.004} yMax={0.25} level={level} fmtValue={f4}
          ariaLabel={`Cp error by mesh. ${sum.lines.map((l) => `${meshLabel(l.mesh)}: FVM ${f4(l.fvm)}, PINN ${f4(l.pinnLo)}${l.pinnHi !== l.pinnLo ? ` to ${f4(l.pinnHi)}` : ''}`).join('; ')}`}
          legend={[{ kind: 'bar', label: 'FVM, same mesh' }, { kind: 'dot', label: 'PINN, positivity on' }, { kind: 'hollow', label: 'positivity off' }, { kind: 'tri', label: 'grid-sequenced' }, { kind: 'dash', label: 'panel method, 140 panels' }]}
          hint="Lower is better. Each PINN is the final round of one seed; hover a mark for its numbers."
        />
      )}
    >
      <ul className="finding-list">
        {sum.lines.map((l) => (
          <li key={l.mesh}>
            <span className="mono">{meshLabel(l.mesh)}</span> PINN <b className="mono">{l.pinnLo === l.pinnHi ? f4(l.pinnLo) : `${f4(l.pinnLo)} to ${f4(l.pinnHi)}`}</b> against FVM <b className="mono">{f4(l.fvm)}</b>
          </li>
        ))}
      </ul>
      <p className="finding-note">One seed per cell: a measured result, not yet a statistical one. The FVM at 400×200 is already at {f4(study.groups.at(-1)?.fvm)}; no PINN was trained there.</p>
    </Finding>
  );
}

function TrajectoryFinding({ rows, e1, error }) {
  const tr = useMemo(() => trajectories(e1, rows), [e1, rows]);
  const sum = useMemo(() => trajectorySummary(e1), [e1]);
  const series = useMemo(() => tr.series.map((s) => ({
    ys: s.ys,
    color: s.positivity ? '--up' : '--lo',
    dash: s.seed === 0 ? [] : s.seed === 1 ? [6, 3] : [2, 3],
    width: 1.7,
    label: s.label,
  })), [tr]);
  const legend = useMemo(() => [
    { label: 'positivity on', color: '--up' }, { label: 'positivity off', color: '--lo' },
  ], []);
  const band = useMemo(() => ({ ...tr.band, label: 'STOP HERE' }), [tr]);
  const hline = useMemo(() => (tr.fvm50 != null ? { y: tr.fvm50, label: `FVM 50×25 ${f4(tr.fvm50)}` } : null), [tr.fvm50]);
  const height = useCallback((w) => Math.max(210, Math.min(250, w * 0.45)), []);
  if (!e1 || !sum) {
    return <MissingFinding eyebrow="E1 · Training trajectory" what={`The round-by-round data${error ? ` (${error})` : ''}`} />;
  }
  return (
    <Finding
      id="finding-trajectory" eyebrow="E1 · Training trajectory"
      title="Accurate by round 3, then most runs drift"
      tag="measured" source="results/e1_rounds.csv"
      chart={(
        <LineChart
          title="Cp RMS by training round, six 50×25 runs (log scale)" label="Cp RMS against FVM 800×400 for each round of six 50×25 runs"
          xs={tr.xs} series={series} legend={legend} logY band={band} hline={hline} height={height}
          note="Seeds 0, 1, 2: solid, dashed, dotted. Each round is one optimiser pass."
        />
      )}
    >
      <p>
        At round 3 all {sum.total} runs sit at <b className="mono">{f4(sum.r3[0])} to {f4(sum.r3[1])}</b>. Then <b>{sum.drifted} of {sum.total}</b> drift, to as much as <b className="mono">{f4(sum.worstFinal)}</b> by the last round, while the training loss keeps falling.
      </p>
      <p className="finding-note">Positivity on or off made no consistent difference. The practice that works is a fixed early stop at rounds 3 to 5; it is not risk-free (one run was already at 0.101 in round 4).</p>
    </Finding>
  );
}

function CylinderFinding({ rows }) {
  const bars = useMemo(() => cylinderBars(rows), [rows]);
  const groups = useMemo(() => bars.map((b) => ({ id: b.id, label: b.label, short: b.short, bar: { value: b.value, kind: b.kind, detail: `${b.detail}: Cp RMS ${f4(b.value)} against FVM 800×400 (measured)` } })), [bars]);
  const v = Object.fromEntries(bars.map((b) => [b.id, b.value]));
  const fvm200 = rows?.find((r) => r.experiment === 'E5' && r.method === 'fvm' && r.mesh === '200x100')?.rms_cp;
  if (bars.length < 4) return <MissingFinding eyebrow="E5 · The cylinder" what="The cylinder rows" />;
  const pinnWins = Math.max(v.pinn, v.pinn_nopos) < v.panel_pg;
  return (
    <Finding
      id="finding-cylinder" eyebrow="E5 · The cylinder"
      title={pinnWins ? 'Where simple physics breaks, the PINN does better' : 'The cylinder at Mach 0.4'}
      tag="measured" source="summary.csv, E5"
      chart={(
        <BarDotChart
          title="Cp RMS against FVM 800×400, cylinder at Mach 0.4 (lower is better)" groups={groups} yMin={0} yMax={0.35} fmtValue={f4} height={200}
          legend={[{ kind: 'bar', label: 'panel-method models' }, { kind: 'bar-pinn', label: 'PINN, 50×25, 10 rounds' }]}
          hint="Hover a bar for its numbers."
        />
      )}
    >
      <p>
        Compressibility matters at Mach 0.4. The incompressible circle is off by <b className="mono">{f4(v.circle)}</b>, the Prandtl–Glauert panel method by <b className="mono">{f4(v.panel_pg)}</b>, the PINN by <b className="mono">{f4(Math.min(v.pinn, v.pinn_nopos))} to {f4(Math.max(v.pinn, v.pinn_nopos))}</b>.
      </p>
      <p className="finding-note">Not a win over the finite-volume solver on a finer mesh{fvm200 != null ? `: the FVM at 200×100 has ${f4(fvm200)}` : ''}. One seed per mode.</p>
    </Finding>
  );
}

function MissingFinding({ eyebrow, what }) {
  return (
    <article className="finding card card--evidence finding--missing">
      <p className="eyebrow">{eyebrow}</p>
      <div className="placeholder">{what} are not available from this server. Nothing is shown on purpose: there is no fake data on this page.</div>
    </article>
  );
}

function Failures({ rows, e1, onNumbers }) {
  const items = useMemo(() => failureItems(rows, e1), [rows, e1]);
  return (
    <section className="fail" aria-labelledby="fail-title">
      <div className="fail-head">
        <p className="eyebrow">Stated plainly</p>
        <h3 id="fail-title" className="card-title">What did not work</h3>
      </div>
      <ul className="fail-list">
        {items.map((it) => (
          <li key={it.id} className="fail-item">
            <p className="fail-n">{it.value}</p>
            <p className="fail-t">{it.title}</p>
            <p className="fail-x">{it.text}</p>
            <p className="src">{it.source}</p>
            {it.link.href ? (
              <a className="arrow-link" href={it.link.href}>{it.link.label} <span aria-hidden="true">→</span></a>
            ) : (
              <button type="button" className="arrow-link arrow-link--button" onClick={() => onNumbers(it.link.exp)}>{it.link.label} <span aria-hidden="true">→</span></button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * The Results page body: headline numbers, three findings with one chart each, what did not work, the speed and cost cards, then the
 * full table and the "what is real" note folded away. Every number is computed from `rows` (GET /api/results) and the E1 round data.
 */
export default function ResultsPage({ rows, error, result }) {
  const { data: e1, error: e1Error } = useRoundsE1();
  const caps = useCapabilities();
  const [open, setOpen] = useState(false);
  const [focus, setFocus] = useState({ exp: 'all', n: 0 });
  const showNumbers = useCallback((exp) => {
    setFocus((f) => ({ exp, n: f.n + 1 }));
    setOpen(true);
    requestAnimationFrame(() => document.getElementById('numbers')?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }, []);

  if (error) {
    return <div className="placeholder">The results are not available from this server ({error}). Nothing is shown on purpose: there is no fake data on this page.</div>;
  }
  if (!rows) return <div className="status" role="status">Loading results…</div>;

  return (
    <div className="results-page">
      <StatBand rows={rows} />
      {caps.fvm_curves === false ? <FvmWithheldNote>Every number on this page is unchanged by that.</FvmWithheldNote> : <FvmCredit />}
      <div className="findings">
        <MeshFinding rows={rows} />
        <TrajectoryFinding rows={rows} e1={e1} error={e1Error} />
        <CylinderFinding rows={rows} />
      </div>
      <Failures rows={rows} e1={e1} onNumbers={showNumbers} />
      <div className="pair">
        <SpeedCard rows={rows} result={result} />
        <TradeoffCard />
      </div>
      <details className="fold" id="numbers" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary><span>All {rows.length} rows of the numbers dashboard</span><span className="fold-hint">filter by experiment or source</span></summary>
        <div className="fold-body"><ResultsTable rows={rows} focus={focus} /></div>
      </details>
      <RealFold />
    </div>
  );
}
