import { useRef, useState } from 'react';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';
import { logTicks125, niceTicks } from '../utils/ticks.js';

const FILL = { fvm: '--muted', model: '--muted', pinn: '--up' };
const rgb = (name) => cssVar(name);

/**
 * Bars and dots on one value axis, drawn with the theme's CSS variables (nothing hard-coded). One group per x slot:
 * `{id, label (string or lines), sub, bar: {value, kind: 'fvm' | 'model' | 'pinn', detail}, dots: [{value, positivity, gridSeq, detail}]}`.
 * `log` plots log10 of the value (the axis is `yMin` to `yMax`). `level` is `{value, label}`: a dashed reference line.
 */
function draw(g, w, h, { groups, log, yMin, yMax, level, fmtValue, layout }) {
  const narrow = (w - 52) / groups.length < 92; // slots too narrow for the long labels: use the short ones
  const labelOf = (gr) => [].concat(narrow && gr.short ? gr.short : gr.label);
  const lines = Math.max(1, ...groups.map((gr) => labelOf(gr).length + (gr.sub ? 1 : 0)));
  const M = { l: 44, r: 8, t: 16, b: 12 + lines * 14 };
  const pw = w - M.l - M.r;
  const ph = h - M.t - M.b;
  const t = (v) => (log ? Math.log10(v) : v);
  const Y = (v) => M.t + (1 - (t(Math.max(v, yMin)) - t(yMin)) / (t(yMax) - t(yMin))) * ph;
  const sw = pw / groups.length;
  const font = cssVar('--font-body');
  const mono = cssVar('--font-mono');

  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  g.lineWidth = 1;
  g.font = `11px ${mono}`;
  const ticks = log ? logTicks125(Math.log10(yMin), Math.log10(yMax)) : niceTicks(yMin, yMax, 4).map((v) => [v, String(v)]);
  ticks.forEach(([v, label]) => {
    g.strokeStyle = cssVar('--grid');
    g.beginPath();
    g.moveTo(M.l, Math.round(Y(v)) + 0.5);
    g.lineTo(w - M.r, Math.round(Y(v)) + 0.5);
    g.stroke();
    g.fillStyle = cssVar('--muted');
    g.textAlign = 'right';
    g.fillText(label, M.l - 6, Y(v) + 4);
  });
  g.strokeStyle = cssVar('--line');
  g.beginPath();
  g.moveTo(M.l, Math.round(h - M.b) + 0.5);
  g.lineTo(w - M.r, Math.round(h - M.b) + 0.5);
  g.stroke();

  const targets = [];
  const labelDots = sw >= 100;
  groups.forEach((gr, i) => {
    const x0 = M.l + i * sw;
    const hasDots = gr.dots && gr.dots.length > 0;
    const bw = Math.min(hasDots ? 40 : 56, sw * (hasDots ? 0.34 : 0.52));
    const bx = hasDots ? x0 + sw * 0.12 : x0 + (sw - bw) / 2;
    if (gr.bar) {
      const top = Y(gr.bar.value);
      const c = rgb(FILL[gr.bar.kind] ?? '--muted');
      g.globalAlpha = gr.bar.kind === 'pinn' ? 0.9 : 0.3;
      g.fillStyle = c;
      g.fillRect(bx, top, bw, h - M.b - top);
      g.globalAlpha = 1;
      g.fillStyle = c;
      g.fillRect(bx, top, bw, 2.5);
      g.fillStyle = cssVar('--ink');
      g.font = `500 11px ${mono}`;
      g.textAlign = 'center';
      g.fillText(fmtValue(gr.bar.value), bx + bw / 2, top - 6);
      targets.push({ px: bx + bw / 2, py: (top + h - M.b) / 2, rx: bw / 2 + 6, ry: (h - M.b - top) / 2 + 8, detail: gr.bar.detail });
    }
    if (hasDots) {
      const n = gr.dots.length;
      const cx = x0 + sw * 0.7;
      const spread = Math.min(12, (sw * 0.3) / Math.max(1, n - 1));
      const placed = gr.dots.map((d, k) => ({ d, px: cx + (k - (n - 1) / 2) * spread, py: Y(d.value) }));
      placed.forEach(({ d, px, py }) => {
        const c = rgb('--up');
        g.strokeStyle = c;
        g.lineWidth = 2;
        g.fillStyle = d.positivity ? c : cssVar('--plot-bg');
        g.beginPath();
        if (d.gridSeq) {
          g.moveTo(px, py - 6.5);
          g.lineTo(px + 6, py + 5);
          g.lineTo(px - 6, py + 5);
          g.closePath();
        } else g.arc(px, py, 5, 0, 6.2832);
        g.fill();
        g.stroke();
        targets.push({ px, py, rx: 9, ry: 9, detail: d.detail });
      });
      // value labels: a lone dot gets its value on its right; dots within 16 px of each other share one range label above the cluster
      if (labelDots) {
        const sorted = [...placed].sort((a, b) => a.py - b.py);
        const clusters = [];
        sorted.forEach((p) => {
          const last = clusters[clusters.length - 1];
          if (last && p.py - last[last.length - 1].py < 16) last.push(p);
          else clusters.push([p]);
        });
        g.fillStyle = rgb('--up');
        g.font = `500 11px ${mono}`;
        clusters.forEach((cl) => {
          const vals = cl.map((p) => p.d.value);
          const lo = Math.min(...vals);
          const hi = Math.max(...vals);
          if (cl.length === 1) {
            g.textAlign = 'left';
            g.fillText(fmtValue(lo), cl[0].px + 10, cl[0].py + 4);
          } else {
            g.textAlign = 'center';
            g.fillText(`${fmtValue(lo)}–${fmtValue(hi)}`, cx, cl[0].py - 12);
          }
        });
      }
    }
    g.fillStyle = cssVar('--ink');
    g.font = `500 12px ${font}`;
    g.textAlign = 'center';
    const ls = labelOf(gr);
    ls.forEach((l, k) => g.fillText(l, x0 + sw / 2, h - M.b + 16 + k * 14));
    if (gr.sub) {
      g.fillStyle = cssVar('--muted');
      g.font = `11px ${font}`;
      g.fillText(gr.sub, x0 + sw / 2, h - M.b + 16 + ls.length * 14);
    }
  });

  if (level) {
    g.save();
    g.setLineDash([5, 4]);
    g.strokeStyle = cssVar('--lo');
    g.lineWidth = 1.4;
    g.beginPath();
    g.moveTo(M.l, Y(level.value));
    g.lineTo(w - M.r, Y(level.value));
    g.stroke();
    g.restore();
    g.fillStyle = cssVar('--lo');
    g.font = `500 10.5px ${mono}`;
    if (narrow) {
      g.textAlign = 'left';
      g.fillText(level.short ?? level.label, M.l + 3, Y(level.value) - 5);
    } else {
      g.textAlign = 'right';
      g.fillText(level.label, w - M.r - 2, Y(level.value) - 5);
    }
  }
  layout.current = targets;
}

/** The legend marks: bar, dot, hollow dot, triangle or dashed line, each in a theme colour. */
function Swatch({ kind }) {
  return <i className={`sw sw--${kind}`} aria-hidden="true" />;
}

export default function BarDotChart({ title, ariaLabel, groups, log = false, yMin, yMax, level = null, fmtValue, legend = [], height = 215, hint }) {
  const ref = useRef(null);
  const layout = useRef([]);
  const [hover, setHover] = useState('');
  useCanvasDraw(ref, (w) => Math.max(height - 30, Math.min(height, w * 0.62)), (g, w, h) => draw(g, w, h, { groups, log, yMin, yMax, level, fmtValue, layout }), [groups, log, yMin, yMax, level]);

  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const hit = layout.current.find((p) => Math.abs(p.px - x) <= p.rx && Math.abs(p.py - y) <= p.ry);
    setHover(hit ? hit.detail : '');
  };
  return (
    <figure className="barchart">
      <figcaption className="k">{title}</figcaption>
      <canvas ref={ref} role="img" aria-label={ariaLabel ?? title} onMouseMove={onMove} onMouseLeave={() => setHover('')} />
      <div className="legend">
        {legend.map((l) => (
          <span key={l.label}><Swatch kind={l.kind} />{l.label}</span>
        ))}
      </div>
      <p className={`hint barchart-read${hover ? ' has-hover' : ''}`} role="status" aria-live="polite">{hover || hint || 'Hover a mark for its numbers.'}</p>
    </figure>
  );
}
