import { useCallback, useEffect, useRef } from 'react';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';
import { canvasPalette } from '../theme/canvasPalette.js';
import { currentTheme } from '../theme/theme.js';
import { useDrawPointer, WINDOW } from '../hooks/useDrawPointer.js';
import { FlowCard } from './ControlCards.jsx';

const ASPECT = 1.8;

// ---- the pen layer: a transparent canvas over the flow picture --------------------------------------

function drawPen(g, w, h, stroke, { mirror, hint }) {
  g.clearRect(0, 0, w, h);
  const T = canvasPalette(currentTheme());
  const pen = (a) => `rgba(${T.pen.join(',')},${a})`;
  const X = (x) => ((x - WINDOW.xmin) / (WINDOW.xmax - WINDOW.xmin)) * w;
  const Y = (y) => ((WINDOW.ymax - y) / (WINDOW.ymax - WINDOW.ymin)) * h;
  if (mirror) {
    g.save();
    g.setLineDash([6, 5]);
    g.strokeStyle = pen(0.7);
    g.lineWidth = 1.2;
    g.beginPath();
    g.moveTo(0, Y(0));
    g.lineTo(w, Y(0));
    g.stroke();
    g.setLineDash([]);
    g.font = '500 11px "IBM Plex Mono", monospace';
    g.fillStyle = pen(0.85);
    g.fillText('MIRROR LINE: DRAW ABOVE', 10, Y(0) - 7);
    g.restore();
  }
  if (!stroke && hint) {
    g.font = '500 12px "IBM Plex Mono", monospace';
    g.textAlign = 'center';
    g.fillStyle = cssVar('--muted');
    g.fillText(hint, w / 2, h / 2);
    g.textAlign = 'start';
  }
  if (!stroke || stroke.length < 2) return;
  const trace = (pts) => {
    g.beginPath();
    pts.forEach((q, i) => (i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]))));
  };
  g.lineCap = 'round';
  g.lineJoin = 'round';
  if (mirror) {
    g.strokeStyle = pen(0.35);
    g.lineWidth = 2;
    trace(stroke.map((q) => [q[0], -Math.max(0, q[1])]));
    g.stroke();
  }
  const shown = mirror ? stroke.map((q) => [q[0], Math.max(0, q[1])]) : stroke;
  g.strokeStyle = T.penHalo;
  g.lineWidth = 5;
  trace(shown);
  g.stroke();
  g.strokeStyle = pen(1);
  g.lineWidth = 2.4;
  trace(shown);
  g.stroke();
  if (!mirror) {
    // the outline will be closed back to the start: show that as a faint dashed line
    g.setLineDash([3, 5]);
    g.strokeStyle = pen(0.5);
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(X(stroke[stroke.length - 1][0]), Y(stroke[stroke.length - 1][1]));
    g.lineTo(X(stroke[0][0]), Y(stroke[0][1]));
    g.stroke();
    g.setLineDash([]);
  }
}

/**
 * Pointer drawing on top of the flow picture. A mouse or pen always draws; a finger draws only while `armed`.
 * `onStroke(points)` is called once per finished stroke with the raw world points.
 */
export function PenLayer({ armed, mirror, hint, onStroke }) {
  const ref = useRef(null);
  const strokeRef = useRef(null);
  const optsRef = useRef({ mirror, hint });
  optsRef.current = { mirror, hint };

  const paint = useCallback(() => {
    const c = ref.current;
    if (!c || !c.clientWidth) return;
    const g = c.getContext('2d');
    const k = c.width / c.clientWidth;
    g.setTransform(k, 0, 0, k, 0, 0);
    drawPen(g, c.clientWidth, c.clientHeight, strokeRef.current, optsRef.current);
  }, []);

  useCanvasDraw(ref, (w) => Math.round(w / ASPECT), (g, w, h) => drawPen(g, w, h, strokeRef.current, optsRef.current), [mirror, hint]);
  useDrawPointer(ref, armed, {
    onLive: (pts) => {
      strokeRef.current = pts;
      paint();
    },
    onComplete: onStroke,
  });
  useEffect(() => {
    paint();
  }, [paint, mirror, hint]);

  return <canvas ref={ref} className={`flow-overlay pen${armed ? ' armed' : ''}`} aria-label="Drawing area" />;
}

// ---- controls -------------------------------------------------------------------------------------

/** Small outline preview of a library shape, scaled to fit. */
export function ShapeIcon({ points, ground }) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const W = 72;
  const H = 34;
  const s = Math.min((W - 8) / (x1 - x0 || 1), (H - 8) / (y1 - y0 || 1));
  const ox = (W - (x1 - x0) * s) / 2;
  const oy = (H - (y1 - y0) * s) / 2;
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${(ox + (p[0] - x0) * s).toFixed(1)} ${(H - oy - (p[1] - y0) * s).toFixed(1)}`).join('') + 'Z';
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} aria-hidden="true" focusable="false">
      {ground != null && <line x1="2" x2={W - 2} y1={H - oy + 1} y2={H - oy + 1} stroke="currentColor" strokeOpacity=".5" strokeDasharray="3 2" />}
      <path d={d} fill="currentColor" />
    </svg>
  );
}

export function ShapeCard({ shapes, value, onChange, onUndo, onClear, canUndo }) {
  return (
    <section className="card card--shape">
      <h2>1 · Shape</h2>
      <div className="seg" role="group" aria-label="Drawing tools">
        <button type="button" aria-pressed={value.armed} onClick={() => onChange({ armed: !value.armed })} title="Mouse and pen always draw; this arms your finger on a touch screen">
          ✎ Draw
        </button>
        <button type="button" className="plain" onClick={onUndo} disabled={!canUndo}>Undo</button>
        <button type="button" className="plain" onClick={onClear}>Clear</button>
      </div>
      <label className="check">
        <input type="checkbox" checked={value.mirror} onChange={(e) => onChange({ mirror: e.target.checked })} />
        <span>Mirror about the centre line <span className="hint-inline">(draw the upper half only)</span></span>
      </label>
      <div className="shape-grid" role="group" aria-label="Shape library">
        {shapes.map((s) => (
          <button key={s.id} type="button" className="shape-card" aria-pressed={value.presetId === s.id} onClick={() => onChange({ presetId: s.id })}>
            <ShapeIcon points={s.points} ground={s.ground} />
            <span>{s.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

export function OptionsCard({ value, onChange }) {
  return (
    <section className="card">
      <h2>2 · Physics options</h2>
      <label className="check">
        <input type="checkbox" checked={value.lifting} onChange={(e) => onChange({ lifting: e.target.checked })} />
        <span>Lift (Kutta condition) <span className="hint-inline">off for blunt bodies</span></span>
      </label>
      <label className="check">
        <input type="checkbox" checked={value.groundOn} onChange={(e) => onChange({ groundOn: e.target.checked })} />
        <span>Road below the shape</span>
      </label>
      <div className={value.groundOn ? '' : 'dim'}>
        <label className="field" htmlFor="gap">Gap to the road <span className="val">{value.gap.toFixed(2)} chord</span></label>
        <input type="range" id="gap" min="0.02" max="0.3" step="0.01" value={value.gap} disabled={!value.groundOn} onChange={(e) => onChange({ gap: +e.target.value })} />
      </div>
    </section>
  );
}

export function FilesCard({ onImport, onExport, canExport }) {
  return (
    <section className="card">
      <h2>4 · Files</h2>
      <div className="upload">
        <button type="button" className="btn" onClick={onExport} disabled={!canExport}>Export .dat</button>
      </div>
      <div style={{ marginTop: 8 }}>
        <label htmlFor="datfile" className="hint">Import a UIUC airfoil <code>.dat</code> file (Selig or Lednicer):</label>
        <input type="file" id="datfile" accept=".dat,.txt" onChange={(e) => { onImport(e.target.files[0]); e.target.value = ''; }} />
      </div>
    </section>
  );
}

/**
 * All the Draw-tab controls. `value` is `{armed, mirror, presetId, lifting, groundOn, gap, alpha, mach}`;
 * `onChange(partial)` merges changes into it.
 */
export default function DrawPad({ shapes, value, onChange, onUndo, onClear, canUndo, onImport, onExport, canExport }) {
  return (
    <>
      <ShapeCard shapes={shapes} value={value} onChange={onChange} onUndo={onUndo} onClear={onClear} canUndo={canUndo} />
      <OptionsCard value={value} onChange={onChange} />
      <FlowCard idp="d-" title="3 · Flow conditions" alpha={value.alpha} mach={value.mach} onAlpha={(alpha) => onChange({ alpha })} onMach={(mach) => onChange({ mach })}
        hint="Subsonic only (Ma < 0.7). The drawing is scaled to length 1 before it is solved." />
      <FilesCard onImport={onImport} onExport={onExport} canExport={canExport} />
    </>
  );
}
