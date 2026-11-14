import { useEffect, useMemo, useState } from 'react';
import { fmt, fmtMs, fmtSec, meshLabel } from '../utils/format.js';
import { EXPERIMENTS, methodLabel, rowsFor } from '../utils/pinnCase.js';
import Tag from './Tag.jsx';

const SHOWN = 12; // rows shown before "Show all"
const SOURCES = ['all', 'measured', 'quoted', 'fitted'];

/**
 * The numbers dashboard: every row of results/summary.csv as exported (GET /api/results), filtered by experiment and by source.
 * A number is never shown without its source tag; an empty cell is an en dash.
 */
export default function ResultsTable({ rows, error, focus = null }) {
  const [exp, setExp] = useState('all');
  const [source, setSource] = useState('all');
  const [all, setAll] = useState(false);

  // A link elsewhere on the page ("see the E6 rows") sets the filter: `focus` is {exp, n}, n counts the clicks.
  useEffect(() => {
    if (focus && focus.n > 0) {
      setExp(focus.exp);
      setSource('all');
      setAll(false);
    }
  }, [focus?.n]); // eslint-disable-line react-hooks/exhaustive-deps

  const experiments = useMemo(() => ['all', ...Array.from(new Set((rows ?? []).map((r) => r.experiment))).sort()], [rows]);
  const shown = useMemo(() => rowsFor(rows, exp, source), [rows, exp, source]);

  if (error) {
    return <div className="placeholder">The results table is not available from this server ({error}). Nothing is shown on purpose: there is no fake data on this page.</div>;
  }
  if (!rows) return <div className="status" role="status">Loading results…</div>;

  const visible = all ? shown : shown.slice(0, SHOWN);
  return (
    <div className="numbers">
      <p className="hint" style={{ marginTop: 0 }}>
        All {rows.length} rows of <code>results/summary.csv</code> as exported. <Tag kind="measured" /> produced by code in this repo on this laptop,{' '}
        <Tag kind="quoted" /> copied from the paper (arXiv 2401.08705) or another document, <Tag kind="fitted" /> tuned to data. Cp RMS is against the FVM 800×400 solution
        unless the notes say otherwise.
      </p>
      <div className="filters">
        <div className="seg" role="group" aria-label="Experiment">
          {experiments.map((e) => (
            <button key={e} type="button" aria-pressed={e === exp} onClick={() => { setExp(e); setAll(false); }}>{e === 'all' ? 'All' : e}</button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Source">
          {SOURCES.map((s) => (
            <button key={s} type="button" aria-pressed={s === source} onClick={() => { setSource(s); setAll(false); }}>{s === 'all' ? 'Any source' : s}</button>
          ))}
        </div>
      </div>
      <p className="exp-note" role="status">
        {exp === 'all' ? 'Pick an experiment to see what it tested.' : EXPERIMENTS[exp] ?? 'Rows of this experiment as exported.'}{' '}
        <span className="mono">{shown.length} rows</span>
      </p>
      <div className="table-scroll">
        <table className="results-table compact">
          <thead>
            <tr>
              <th>Exp.</th><th>Method</th><th>Mesh</th><th>Case</th><th className="num">Cp RMS</th><th className="num">Cl (ref.)</th>
              <th className="num">Train</th><th className="num">Per query</th><th>Source</th><th><span className="visually-hidden">Notes</span></th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r, i) => (
              <tr key={`${r.experiment}-${r.case}-${r.method}-${r.mesh}-${i}`}>
                <td className="mono">{r.experiment}</td>
                <td>{methodLabel(r.method)}</td>
                <td className="mono">{meshLabel(r.mesh)}</td>
                <td className="mono case-cell" title={r.case ?? undefined}>{r.case ?? '–'}</td>
                <td className="num mono">{fmt(r.rms_cp, 4)}</td>
                <td className="num mono">{fmt(r.cl, 3)}{r.cl_ref != null && <span className="sub"> ({fmt(r.cl_ref, 3)})</span>}</td>
                <td className="num mono">{fmtSec(r.train_s)}</td>
                <td className="num mono">{fmtMs(r.infer_ms)}</td>
                <td><Tag kind={r.source} /></td>
                <td>{r.notes ? <details className="row-notes"><summary>notes</summary><p>{r.notes}</p></details> : null}</td>
              </tr>
            ))}
            {!visible.length && <tr><td colSpan={10} className="tbd">No rows match these filters.</td></tr>}
          </tbody>
        </table>
      </div>
      {shown.length > SHOWN && (
        <p>
          <button type="button" className="btn" onClick={() => setAll(!all)}>{all ? `Show the first ${SHOWN} rows` : `Show all ${shown.length} rows`}</button>
        </p>
      )}
    </div>
  );
}
