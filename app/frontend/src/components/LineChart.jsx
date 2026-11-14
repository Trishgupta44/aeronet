import { useRef } from 'react';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';
import { decadeLabel, decadeTicks, logTicks125, niceTicks } from '../utils/ticks.js';

const M = { l: 50, r: 12, t: 10, b: 32 };

/**
 * One small canvas chart: lines against the training round. `series` is [{ys, color (a CSS variable name such as '--up'), dash, label}], `xs` the shared x values.
 * `logY` plots log10 (non-positive values are skipped). `hline` is {y, label} for a reference level. `cursor` is an x value
 * (the replay round): a vertical line with a dot on every series. `band` is {from, to, label}: a shaded span of x values (for example the
 * rounds where training should stop), drawn behind the lines.
 */
function draw(g, w, h, { xs, series, logY, hline, cursor, xLabel, band }) {
  g.fillStyle = cssVar('--plot-bg');
  g.fillRect(0, 0, w, h);
  const t = (v) => (logY ? Math.log10(v) : v);
  const ok = (v) => v != null && Number.isFinite(v) && (!logY || v > 0);
  const vals = [];
  series.forEach((s) => s.ys.forEach((v) => ok(v) && vals.push(t(v))));
  if (hline && ok(hline.y)) vals.push(t(hline.y));
  g.font = '11px system-ui';
  g.textAlign = 'center';
  if (!vals.length || !xs.length) {
    g.fillStyle = cssVar('--muted');
    g.fillText('no data', w / 2, h / 2);
    return;
  }
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const pad = (hi - lo || 1) * 0.08;
  lo -= pad;
  hi += pad;
  const x0 = xs[0];
  const x1 = xs[xs.length - 1] === x0 ? x0 + 1 : xs[xs.length - 1];
  const X = (x) => M.l + ((x - x0) / (x1 - x0)) * (w - M.l - M.r);
  const Y = (v) => M.t + (1 - (t(v) - lo) / (hi - lo)) * (h - M.t - M.b);

  if (band) {
    g.fillStyle = cssVar('--heading');
    g.globalAlpha = 0.11;
    g.fillRect(X(band.from), M.t, X(band.to) - X(band.from), h - M.t - M.b);
    g.globalAlpha = 1;
    g.strokeStyle = cssVar('--heading');
    g.lineWidth = 1;
    [band.from, band.to].forEach((x) => {
      g.beginPath();
      g.moveTo(X(x), M.t);
      g.lineTo(X(x), h - M.b);
      g.stroke();
    });
    g.fillStyle = cssVar('--heading');
    g.textAlign = 'center';
    g.font = `500 10.5px ${cssVar('--font-mono')}`;
    g.fillText(band.label, (X(band.from) + X(band.to)) / 2, M.t + 12);
    g.font = '11px system-ui';
  }
  g.fillStyle = cssVar('--muted');
  g.strokeStyle = cssVar('--grid');
  g.lineWidth = 1;
  const decades = decadeTicks(Math.ceil(lo), Math.floor(hi), 5);
  let yTicks;
  if (!logY) yTicks = niceTicks(lo, hi, 4).map((v) => [v, String(v)]);
  else yTicks = decades.length >= 3 ? decades.map((k) => [10 ** k, decadeLabel(k)]) : logTicks125(lo, hi);
  g.textAlign = 'right';
  yTicks.forEach(([v, label]) => {
    g.beginPath();
    g.moveTo(M.l, Y(v));
    g.lineTo(w - M.r, Y(v));
    g.stroke();
    g.fillText(label, M.l - 6, Y(v) + 4);
  });
  g.textAlign = 'center';
  niceTicks(x0, x1, Math.max(2, Math.floor((w - M.l - M.r) / 60))).forEach((v) => {
    if (!Number.isInteger(v)) return;
    g.fillText(String(v), X(v), h - M.b + 15);
  });
  g.fillText(xLabel, (M.l + w - M.r) / 2, h - 4);

  if (hline && ok(hline.y)) {
    g.save();
    g.setLineDash([6, 4]);
    g.strokeStyle = cssVar('--ref');
    g.lineWidth = 1.2;
    g.beginPath();
    g.moveTo(M.l, Y(hline.y));
    g.lineTo(w - M.r, Y(hline.y));
    g.stroke();
    g.restore();
    g.textAlign = 'right';
    g.lineJoin = 'round';
    g.lineWidth = 4;
    g.strokeStyle = cssVar('--plot-bg');
    g.strokeText(hline.label, w - M.r - 4, Y(hline.y) - 5);
    g.fillStyle = cssVar('--ink');
    g.fillText(hline.label, w - M.r - 4, Y(hline.y) - 5);
  }

  series.forEach((s) => {
    g.save();
    g.setLineDash(s.dash || []);
    g.strokeStyle = cssVar(s.color);
    g.lineWidth = s.width || 1.8;
    g.beginPath();
    let pen = false;
    s.ys.forEach((v, i) => {
      if (!ok(v)) {
        pen = false;
        return;
      }
      if (pen) g.lineTo(X(xs[i]), Y(v));
      else g.moveTo(X(xs[i]), Y(v));
      pen = true;
    });
    g.stroke();
    g.restore();
    g.fillStyle = cssVar(s.color);
    s.ys.forEach((v, i) => {
      if (!ok(v)) return;
      g.beginPath();
      g.arc(X(xs[i]), Y(v), 2.2, 0, 6.2832);
      g.fill();
    });
  });

  if (cursor != null) {
    g.strokeStyle = cssVar('--ink');
    g.lineWidth = 1;
    g.setLineDash([2, 3]);
    g.beginPath();
    g.moveTo(X(cursor), M.t);
    g.lineTo(X(cursor), h - M.b);
    g.stroke();
    g.setLineDash([]);
    const i = xs.indexOf(cursor);
    series.forEach((s) => {
      const v = s.ys[i];
      if (i < 0 || !ok(v)) return;
      g.fillStyle = cssVar('--plot-bg');
      g.strokeStyle = cssVar(s.color);
      g.lineWidth = 2;
      g.beginPath();
      g.arc(X(cursor), Y(v), 4.6, 0, 6.2832);
      g.fill();
      g.stroke();
    });
  }
}

export default function LineChart({ title, label, xs, series, logY = false, hline = null, cursor = null, xLabel = 'training round', note, band = null, legend = null, height = null }) {
  const ref = useRef(null);
  useCanvasDraw(ref, height ?? ((w) => Math.max(170, Math.min(230, w * 0.6))), (g, w, h) => draw(g, w, h, { xs, series, logY, hline, cursor, xLabel, band }), [xs, series, logY, hline, cursor, xLabel, band]);
  return (
    <figure className="linechart">
      <figcaption className="k">{title}</figcaption>
      <canvas ref={ref} role="img" aria-label={label ?? title} />
      <div className="legend">
        {(legend ?? series).map((s) => (
          <span key={s.label}><i style={{ background: `var(${s.color})`, borderRadius: 0, height: 3, verticalAlign: 3 }} />{s.label}</span>
        ))}
      </div>
      {note && <p className="hint">{note}</p>}
    </figure>
  );
}
