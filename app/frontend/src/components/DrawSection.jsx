import { useCallback, useEffect, useRef, useState } from 'react';
import { getShapes, parseDat } from '../api.js';
import { useSolve } from '../hooks/useSolve.js';
import { saveText } from '../utils/download.js';
import { processStroke, toDat } from '../utils/outline.js';
import CaseView from './CaseView.jsx';
import DrawPad, { PenLayer } from './DrawPad.jsx';

const UNDO_LIMIT = 10;
const DEFAULT_GAP = 0.1;

/** A library entry as the editable state of the pad. `over` carries values restored from the page address. */
function presetSnap(p, over = {}) {
  const restored = over.lifting !== undefined;
  const ground = restored ? over.ground : p.ground ?? '';
  return {
    id: p.id,
    name: p.label,
    points: p.points,
    note: p.note,
    lifting: restored ? over.lifting : p.lifting,
    groundOn: ground !== '' && ground != null,
    gap: ground !== '' && ground != null ? ground : DEFAULT_GAP,
  };
}

/**
 * The Draw section: pad controls on the left; flow picture, pen layer, results and Cp on the right.
 * `initial` is `{shape, lifting, ground, alpha, mach}` from the page address; `request` (`{id, alpha, mach, nonce}`) lets the
 * page load a library shape from outside (present mode); `onState` reports the state for the address.
 */
export default function DrawSection({ layers, onLayersChange, paused, onPausedChange, initial = {}, request = null, onState, onActivate }) {
  const [shapes, setShapes] = useState([]);
  const [pen, setPen] = useState({ armed: false, mirror: false });
  const [alpha, setAlpha] = useState(initial.alpha ?? 4);
  const [mach, setMach] = useState(initial.mach ?? 0);
  const [hist, setHist] = useState({ cur: null, undo: [] });
  const [msg, setMsg] = useState(null);
  const optsRef = useRef({ lifting: true, groundOn: false, gap: DEFAULT_GAP });
  const cur = hist.cur;
  if (cur) optsRef.current = { lifting: cur.lifting, groundOn: cur.groundOn, gap: cur.gap };

  const input = cur
    ? { kind: 'shape', points: cur.points, alpha, mach, lifting: cur.lifting, ground: cur.groundOn ? cur.gap : null, reference: null }
    : null;
  const { result, error, failedPoints } = useSolve(input);

  const commit = useCallback((snap) => {
    setHist((h) => ({ cur: snap, undo: h.cur ? [...h.undo, h.cur].slice(-UNDO_LIMIT) : h.undo }));
  }, []);
  const undo = () => {
    setMsg(null);
    setHist((h) => (h.undo.length ? { cur: h.undo[h.undo.length - 1], undo: h.undo.slice(0, -1) } : h));
  };
  const clear = () => {
    setMsg(null);
    setHist((h) => ({ cur: null, undo: h.cur ? [...h.undo, h.cur].slice(-UNDO_LIMIT) : h.undo }));
  };

  // library, and the first shape (from the address, else the bird wing)
  const booted = useRef(false);
  useEffect(() => {
    getShapes()
      .then((list) => {
        setShapes(list);
        if (booted.current) return;
        booted.current = true;
        const p = list.find((s) => s.id === initial.shape) ?? list.find((s) => s.id === 'bird') ?? list[0];
        if (p) setHist((h) => (h.cur ? h : { ...h, cur: presetSnap(p, p.id === initial.shape ? initial : {}) }));
      })
      .catch((e) => setMsg({ kind: 'err', text: e.message }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // library shape requested by the page (present mode)
  useEffect(() => {
    if (!request) return;
    const p = shapes.find((s) => s.id === request.id);
    if (!p) return;
    setMsg(null);
    commit(presetSnap(p));
    if (request.alpha !== undefined) setAlpha(request.alpha);
    if (request.mach !== undefined) setMach(request.mach);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request?.nonce, shapes]);

  // a drawing the solver rejects goes back to the previous shape, with the solver's message
  useEffect(() => {
    if (!error || !cur || failedPoints !== cur.points || /^Cannot reach/.test(error)) return;
    setMsg({ kind: 'err', text: error });
    setHist((h) => (h.undo.length ? { cur: h.undo[h.undo.length - 1], undo: h.undo.slice(0, -1) } : { cur: null, undo: [] }));
  }, [error, failedPoints, cur]);

  useEffect(() => {
    if (cur) onState?.({ shape: cur.id, lifting: cur.lifting, ground: cur.groundOn ? Math.round(cur.gap * 100) / 100 : '', alpha, mach });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cur, alpha, mach]);

  const onStroke = (stroke) => {
    onActivate?.();
    const r = processStroke(stroke, { mirror: pen.mirror });
    if (r.error) {
      setMsg({ kind: 'err', text: r.error });
      return;
    }
    setMsg(r.closedGap > 0.08 ? { kind: 'info', text: 'The loop was not closed, so the end was joined to the start.' } : null);
    commit({ id: '', name: 'Hand-drawn shape', points: r.points, note: '', ...optsRef.current });
  };

  const onChange = (patch) => {
    onActivate?.();
    if ('alpha' in patch) setAlpha(patch.alpha);
    if ('mach' in patch) setMach(patch.mach);
    const penPatch = {};
    for (const k of ['armed', 'mirror']) if (k in patch) penPatch[k] = patch[k];
    if (Object.keys(penPatch).length) setPen((p) => ({ ...p, ...penPatch }));
    if ('presetId' in patch) {
      const p = shapes.find((s) => s.id === patch.presetId);
      if (p) {
        setMsg(null);
        commit(presetSnap(p));
      }
    }
    const solverPatch = {};
    for (const k of ['lifting', 'groundOn', 'gap']) if (k in patch) solverPatch[k] = patch[k];
    if (Object.keys(solverPatch).length) setHist((h) => (h.cur ? { ...h, cur: { ...h.cur, ...solverPatch } } : h));
  };

  const onImport = async (file) => {
    if (!file) return;
    onActivate?.();
    try {
      const d = await parseDat(await file.text(), file.name);
      commit({ id: '', name: d.name, points: d.points, note: '', lifting: true, groundOn: false, gap: DEFAULT_GAP });
      setMsg({ kind: 'ok', text: `Imported ${d.name} (${d.points.length} points).` });
    } catch (e) {
      setMsg({ kind: 'err', text: e.message });
    }
  };
  const onExport = () => {
    const pts = result?.code === 'custom' ? result.x.map((x, i) => [x, result.y[i]]) : cur?.points;
    if (pts) saveText(toDat(pts, cur?.name ?? 'AeroNet shape'), 'aeronet-shape.dat');
  };

  const value = {
    ...pen,
    presetId: cur?.id ?? '',
    lifting: cur?.lifting ?? optsRef.current.lifting,
    groundOn: cur?.groundOn ?? optsRef.current.groundOn,
    gap: cur?.gap ?? optsRef.current.gap,
    alpha,
    mach,
  };

  const shown = msg ?? (error ? { kind: 'err', text: error } : null);
  const status = (
    <div className="padstatus" role="status">
      {cur?.note && <p className="pill pill--note">{cur.note}</p>}
      {shown && <p className={`msg msg--${shown.kind}`}>{shown.text}</p>}
      {result?.warnings?.map((w) => (
        <p key={w} className="msg msg--warn">{w}</p>
      ))}
      {!shown && !cur && <p className="msg">Nothing to solve yet. Draw a closed loop on the picture, or pick a shape from the library.</p>}
    </div>
  );

  return (
    <div className="layout" onPointerDownCapture={onActivate}>
      <aside className="stack">
        <DrawPad shapes={shapes} value={value} onChange={onChange} onUndo={undo} onClear={clear} canUndo={hist.undo.length > 0}
          onImport={onImport} onExport={onExport} canExport={!!cur} />
      </aside>
      <CaseView idp="d-" result={result} reference={null} layers={layers} onLayersChange={onLayersChange} paused={paused} onPausedChange={onPausedChange} status={status}>
        <PenLayer armed={pen.armed} mirror={pen.mirror} hint={cur ? '' : 'DRAW A CLOSED SHAPE HERE'} onStroke={onStroke} />
      </CaseView>
    </div>
  );
}
