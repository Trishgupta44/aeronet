import { useEffect, useRef, useState } from 'react';
import { cssVar, useCanvasDraw } from '../hooks/useCanvasDraw.js';
import { cmap, gradientCss } from '../utils/colormap.js';
import { ISO_LEVELS, isoLines } from '../utils/contours.js';
import { sampleField, sampleVelocity } from '../utils/field.js';
import { createParticles, mulberry32, stepParticles } from '../utils/particles.js';

export const DEFAULT_LAYERS = { field: true, isobars: true, particles: true, forces: false };
export const HERO_LAYERS = { field: true, isobars: false, particles: true, forces: false };

const ASPECT = 1.8; // the field window is 1.8 : 1 (see backend field.py)
const DT = 0.006; // world units per frame at speed 1
const offscreen = typeof document !== 'undefined' ? document.createElement('canvas') : null;

export const prefersReducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

function arrow(g, x0, y0, x1, y1, color, halo = true) {
  const len = Math.hypot(x1 - x0, y1 - y0);
  if (len < 2) return;
  const a = Math.atan2(y1 - y0, x1 - x0);
  const head = Math.min(7, len * 0.6);
  const path = () => {
    g.beginPath();
    g.moveTo(x0, y0);
    g.lineTo(x1, y1);
  };
  const tip = () => {
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x1 - head * Math.cos(a - 0.45), y1 - head * Math.sin(a - 0.45));
    g.lineTo(x1 - head * Math.cos(a + 0.45), y1 - head * Math.sin(a + 0.45));
    g.closePath();
  };
  g.lineCap = 'round';
  if (halo) {
    g.strokeStyle = 'rgba(255,255,255,.85)';
    g.lineWidth = 3.6;
    path();
    g.stroke();
    tip();
    g.lineWidth = 2.4;
    g.stroke();
  }
  g.strokeStyle = color;
  g.fillStyle = color;
  g.lineWidth = 1.6;
  path();
  g.stroke();
  tip();
  g.fill();
}

function signedArea(xs, ys) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    const j = (i + 1) % xs.length;
    s += xs[i] * ys[j] - xs[j] * ys[i];
  }
  return s / 2;
}

function drawBase(g, w, h, result, layers, hero, ghost) {
  if (hero) g.clearRect(0, 0, w, h);
  else {
    g.fillStyle = layers.field ? '#f7f7f4' : cssVar('--plot-bg');
    g.fillRect(0, 0, w, h);
  }
  const f = result?.field;
  if (!f) return;
  const px = (x) => ((x - f.xmin) / (f.xmax - f.xmin)) * w;
  const py = (y) => ((f.ymax - y) / (f.ymax - f.ymin)) * h;

  if (layers.field) {
    // colour the grid on an off-screen canvas, then scale it up smoothly
    offscreen.width = f.width;
    offscreen.height = f.height;
    const og = offscreen.getContext('2d');
    const img = og.createImageData(f.width, f.height);
    for (let k = 0; k < f.cp.length; k++) {
      let v = f.cp[k];
      if (v === null) {
        // Cells touching the free flow take the mean of their free neighbours, so the stair-steps of the body mask
        // do not show around the drawn outline. Cells deeper inside the body stay transparent.
        const i = k % f.width;
        const nb = [i > 0 ? f.cp[k - 1] : null, i < f.width - 1 ? f.cp[k + 1] : null, f.cp[k - f.width] ?? null, f.cp[k + f.width] ?? null].filter((x) => x !== null);
        if (!nb.length) continue;
        v = nb.reduce((a, b) => a + b, 0) / nb.length;
      }
      const c = cmap(v);
      img.data[k * 4] = c[0];
      img.data[k * 4 + 1] = c[1];
      img.data[k * 4 + 2] = c[2];
      img.data[k * 4 + 3] = 255;
    }
    og.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = 'high';
    g.drawImage(offscreen, 0, 0, w, h);
  }

  if (layers.isobars) {
    g.strokeStyle = layers.field ? 'rgba(0,0,0,.28)' : cssVar('--muted');
    g.lineWidth = 0.8;
    g.lineCap = 'butt';
    for (const level of ISO_LEVELS) {
      g.setLineDash(level < 0 ? [4, 3] : []);
      g.beginPath();
      for (const line of isoLines(f, level)) {
        line.forEach(([x, y], i) => (i ? g.lineTo(px(x), py(y)) : g.moveTo(px(x), py(y))));
      }
      g.stroke();
    }
    g.setLineDash([]);
  }

  if (result.ground_y != null) {
    const y0 = Math.max(0, py(result.ground_y));
    g.save();
    g.beginPath();
    g.rect(0, y0, w, h - y0);
    g.clip();
    g.fillStyle = hero ? 'rgba(10,26,92,.9)' : '#d6d7d3';
    g.fillRect(0, y0, w, h - y0);
    g.strokeStyle = 'rgba(60,66,78,.55)';
    g.lineWidth = 1;
    g.beginPath();
    for (let x = -h; x < w + h; x += 10) {
      g.moveTo(x, y0 + h);
      g.lineTo(x + h, y0);
    }
    g.stroke();
    g.restore();
    g.strokeStyle = '#20242c';
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(0, y0);
    g.lineTo(w, y0);
    g.stroke();
    if (!hero) {
      g.font = '500 11px "IBM Plex Mono", monospace';
      g.fillStyle = '#3c424e';
      g.fillText('ROAD', 10, Math.min(h - 6, y0 + 16));
    }
  }

  g.beginPath();
  result.x.forEach((x, i) => (i ? g.lineTo(px(x), py(result.y[i])) : g.moveTo(px(x), py(result.y[i]))));
  g.closePath();
  g.fillStyle = '#20242c';
  g.fill();
  g.strokeStyle = hero ? 'rgba(255,255,255,.35)' : '#0c0e12';
  g.lineWidth = 1.2;
  g.stroke();

  if (hero) return;

  if (ghost) {
    // the pinned shape as a dashed outline, so a change of shape or angle can be compared at a glance
    g.save();
    g.setLineDash([6, 4]);
    g.strokeStyle = '#0b1020';
    g.lineWidth = 1.4;
    g.beginPath();
    ghost.x.forEach((x, i) => (i ? g.lineTo(px(x), py(ghost.y[i])) : g.moveTo(px(x), py(ghost.y[i]))));
    g.closePath();
    g.stroke();
    g.restore();
  }

  if (layers.forces) {
    const ccw = signedArea(result.x, result.y) > 0;
    const s = result.surface;
    const n = result.x.length;
    const up = cssVar('--up');
    const lo = cssVar('--lo');
    for (let i = 0; i < n; i += 3) {
      const j = (i + 1) % n;
      const dx = result.x[j] - result.x[i];
      const dy = result.y[j] - result.y[i];
      const len = Math.hypot(dx, dy) || 1;
      const tx = dx / len;
      const ty = dy / len;
      const nx = ccw ? ty : -ty; // outward normal
      const ny = ccw ? -tx : tx;
      const L = -s.cp[i] * 0.06; // suction points outward, pressure inward
      arrow(g, px(s.cx[i]), py(s.cy[i]), px(s.cx[i] + nx * L), py(s.cy[i] + ny * L), s.upper[i] ? up : lo);
    }
    const a = (result.alpha * Math.PI) / 180;
    const Ll = 0.15 * result.cl;
    const lx = 0.25; // base of the lift arrow: the quarter-chord point
    arrow(g, px(lx), py(0), px(lx - Math.sin(a) * Ll), py(Math.cos(a) * Ll), '#0b1020');
    g.font = '600 12px "IBM Plex Mono", monospace';
    g.fillStyle = '#0b1020';
    g.fillText('LIFT', px(lx - Math.sin(a) * Ll) + 6, py(Math.cos(a) * Ll) + (Ll >= 0 ? -4 : 12));
  }

  // freestream arrow, tilted by the angle of attack
  const a = (result.alpha * Math.PI) / 180;
  const ax = 34;
  const ay = h / 2;
  const L = 44;
  arrow(g, ax, ay, ax + L * Math.cos(a), ay - L * Math.sin(a), '#20242c', false);
  g.font = '500 11px "IBM Plex Mono", monospace';
  g.fillStyle = '#20242c';
  if (w >= 500) g.fillText('FREESTREAM', ax - 8, ay + 30); // no room for the word on a phone
}

// Trails: one thin stroke per particle, tinted by local speed. Slow air is pale, fast air is deep.
const SPEED_BINS = 8;
const SPEED_LO = 0.85;
const SPEED_HI = 1.5;
const PALETTES = {
  // on the pale colour map: pale blue-grey -> deep navy/indigo
  light: { slow: [150, 166, 196, 0.5], fast: [22, 18, 112, 0.95] },
  // on the dark hero (or a dark plot): blue-white -> white
  dark: { slow: [176, 196, 240, 0.28], fast: [255, 255, 255, 0.85] },
};
const trailStyle = (palette, bin) => {
  const f = bin / (SPEED_BINS - 1);
  const a = palette.slow;
  const b = palette.fast;
  return [0, 1, 2, 3].map((i) => a[i] + f * (b[i] - a[i]));
};

/** Is the picture behind the particles dark? (the hero, or the plain plot background in dark mode) */
function darkBackdrop(hero, fieldOn) {
  if (hero) return true;
  if (fieldOn) return false;
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(cssVar('--plot-bg'));
  return !!m && (0.299 * parseInt(m[1], 16) + 0.587 * parseInt(m[2], 16) + 0.114 * parseInt(m[3], 16)) < 110;
}

/** Draw the latest step of every particle as a 1.15 px round-capped segment; `strength` scales the alpha (still pictures use it for a smooth fade). */
function drawTrails(g, ps, f, w, h, dark, strength = 1) {
  const X = (x) => ((x - f.xmin) / (f.xmax - f.xmin)) * w;
  const Y = (y) => ((f.ymax - y) / (f.ymax - f.ymin)) * h;
  const palette = dark ? PALETTES.dark : PALETTES.light;
  const bins = Array.from({ length: SPEED_BINS }, () => []);
  for (const p of ps) {
    if (p.px === p.x && p.py === p.y) continue;
    const bin = Math.max(0, Math.min(SPEED_BINS - 1, Math.floor(((p.sp - SPEED_LO) / (SPEED_HI - SPEED_LO)) * SPEED_BINS)));
    bins[bin].push(p);
  }
  g.lineCap = 'round';
  g.lineWidth = 1.15;
  bins.forEach((list, bin) => {
    if (!list.length) return;
    const [r, gr, b, a] = trailStyle(palette, bin);
    g.strokeStyle = `rgba(${r | 0},${gr | 0},${b | 0},${(a * strength).toFixed(3)})`;
    g.beginPath();
    for (const p of list) {
      g.moveTo(X(p.px), Y(p.py));
      g.lineTo(X(p.x), Y(p.y));
    }
    g.stroke();
  });
}

/**
 * The layered picture: a base canvas (colour map, isobars, road, body, forces) and an overlay canvas with the animated
 * particles. `onProbe` (panel use only) receives pointer events over the field.
 */
export function FlowStage({ result, layers, paused, hero = false, onProbe, idp = '', ghost = null, stageRef = null }) {
  const baseRef = useRef(null);
  const overlayRef = useRef(null);
  const fieldRef = useRef(null);
  const sizeRef = useRef({ w: 0, h: 0 });
  const particlesRef = useRef([]);
  const randRef = useRef(mulberry32(2024));
  const field = result?.field ?? null;
  fieldRef.current = field;

  const [size, setSize] = useState({ w: 0, h: 0 });
  const getHeight = (w) => Math.round(w / ASPECT);
  useCanvasDraw(baseRef, getHeight, (g, w, h) => drawBase(g, w, h, result, layers, hero, ghost), [result, layers.field, layers.isobars, layers.forces, hero, ghost]);
  useCanvasDraw(overlayRef, getHeight, (g, w, h) => {
    sizeRef.current = { w, h };
    g.clearRect(0, 0, w, h);
    setSize({ w, h });
  }, []);

  if (stageRef) {
    // the picture as one image: colour map and body from the base canvas, the particles from the overlay
    stageRef.current = {
      toBlob: (cb) => {
        const b = baseRef.current;
        const o = overlayRef.current;
        if (!b || !o) return cb(null);
        const c = document.createElement('canvas');
        c.width = b.width;
        c.height = b.height;
        const g = c.getContext('2d');
        g.drawImage(b, 0, 0);
        g.drawImage(o, 0, 0, b.width, b.height);
        return c.toBlob(cb, 'image/png');
      },
    };
  }

  const hasField = !!field;
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return undefined;
    const g = canvas.getContext('2d');
    const { w, h } = size;
    if (!w || !hasField || !layers.particles) {
      g.clearRect(0, 0, w || canvas.width, h || canvas.height);
      return undefined;
    }
    const rand = randRef.current;
    const target = () => {
      const { w: cw, h: ch } = sizeRef.current;
      return Math.max(100, Math.min(hero ? 500 : 700, Math.round(cw * ch * 0.00095 * (hero ? 0.6 : 1))));
    };
    const ensure = () => {
      const f = fieldRef.current;
      const n = target();
      const ps = particlesRef.current;
      if (ps.length < n) particlesRef.current = ps.concat(createParticles(n - ps.length, f, rand));
      else if (ps.length > n) particlesRef.current = ps.slice(0, n);
    };

    if (paused) {
      // a still picture: run the particles a little and keep every trail (also what reduced-motion users get)
      g.clearRect(0, 0, w, h);
      ensure();
      const f = fieldRef.current;
      const dark = darkBackdrop(hero, layers.field);
      for (let k = 0; k < 24; k++) {
        stepParticles(particlesRef.current, f, DT * 1.5, rand);
        drawTrails(g, particlesRef.current, f, w, h, dark, (k + 1) / 24);
      }
      return undefined;
    }

    const dark = darkBackdrop(hero, layers.field);
    let raf = 0;
    let last = performance.now();
    let ema = 0;
    let visible = true;
    const io = typeof IntersectionObserver !== 'undefined' ? new IntersectionObserver(([e]) => { visible = e.isIntersecting; }) : null;
    io?.observe(canvas);
    const frame = (now) => {
      raf = requestAnimationFrame(frame);
      const dtms = now - last;
      last = now;
      if (!visible || document.hidden || dtms <= 0 || dtms > 1000) return;
      const f = fieldRef.current;
      if (!f) return;
      ema = ema ? ema * 0.9 + dtms * 0.1 : dtms;
      window.__aeronetFps = 1000 / ema;
      const k = Math.min(2.5, Math.max(0.5, dtms / 16.7)); // same speed on 60 and 144 Hz screens
      ensure();
      stepParticles(particlesRef.current, f, DT * k, rand);
      g.globalCompositeOperation = 'destination-out';
      g.fillStyle = `rgba(0,0,0,${Math.min(0.3, 0.1 * k)})`; // exponential fade: each trail dims smoothly along its length
      g.fillRect(0, 0, w, h);
      g.globalCompositeOperation = 'source-over';
      drawTrails(g, particlesRef.current, f, w, h, dark);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      io?.disconnect();
    };
  }, [size, hasField, layers.particles, layers.field, paused, hero]);

  const handle = (e) => {
    if (!onProbe) return;
    const f = fieldRef.current;
    if (!f || e.type === 'pointerleave') return onProbe(null);
    const r = e.currentTarget.getBoundingClientRect();
    const cx = e.clientX - r.left;
    const cy = e.clientY - r.top;
    const x = f.xmin + (cx / r.width) * (f.xmax - f.xmin);
    const y = f.ymax - (cy / r.height) * (f.ymax - f.ymin);
    const cp = sampleField(f, 'cp', x, y);
    const v = sampleVelocity(f, x, y);
    return onProbe(cp === null || !v ? null : { left: cx, top: cy, cp, speed: Math.hypot(v[0], v[1]), x });
  };

  return (
    <>
      <canvas ref={baseRef} className="flow-base" id={hero ? undefined : `${idp}field`} role={hero ? undefined : 'img'}
        aria-label={hero ? undefined : 'Pressure coefficient field around the shape'} aria-hidden={hero ? 'true' : undefined} />
      <canvas ref={overlayRef} className="flow-overlay" aria-hidden="true" onPointerMove={handle} onPointerLeave={handle} />
    </>
  );
}

/** The explorer's field card: layer toggles, the stage, hover probe, colour bar and pause / fps controls. */
export default function FlowCanvas({ result, layers, onLayersChange, paused, onPausedChange, idp = '', children, footer, ghost = null, stageRef = null }) {
  const [probe, setProbe] = useState(null);
  const [fps, setFps] = useState(null);

  useEffect(() => {
    const id = setInterval(() => setFps(paused || !window.__aeronetFps ? null : Math.round(window.__aeronetFps)), 500);
    return () => clearInterval(id);
  }, [paused]);

  const toggles = [
    ['field', 'Pressure'],
    ['isobars', 'Isobars'],
    ['particles', 'Flow'],
    ['forces', 'Forces'],
  ];

  return (
    <section className="card">
      <h2>Pressure field (Cp)</h2>
      <div className="flow-controls">
        <div className="seg" role="group" aria-label="Layers">
          {toggles.map(([key, label]) => (
            <button key={key} type="button" aria-pressed={layers[key]} onClick={() => onLayersChange({ ...layers, [key]: !layers[key] })}>
              {label}
            </button>
          ))}
        </div>
        <div className="flow-run">
          {fps !== null && layers.particles && <span className="fps" id={`${idp}fps`} title="Animation frames per second">{fps} fps</span>}
          <button type="button" className="btn" aria-pressed={paused} onClick={() => onPausedChange(!paused)}>{paused ? 'Play' : 'Pause'}</button>
        </div>
      </div>
      <div className="canvas-box flow-box">
        <FlowStage result={result} layers={layers} paused={paused} onProbe={children ? undefined : setProbe} idp={idp} ghost={ghost} stageRef={stageRef} />
        {children}
        {probe && (
          <div className="probe" style={{ left: Math.min(probe.left + 14, 9999), top: probe.top + 14 }} role="status">
            <span>Cp</span><b>{probe.cp.toFixed(2)}</b>
            <span>|V|/V∞</span><b>{probe.speed.toFixed(2)}</b>
            <span>x/c</span><b>{probe.x.toFixed(2)}</b>
          </div>
        )}
      </div>
      <div className="cbar">
        <span>−2 suction</span>
        <div className="bar" id={`${idp}bar`} style={{ background: gradientCss() }} />
        <span>+1 stagnation</span>
      </div>
      <div className="hint">Idealized potential flow (inviscid, no separation or wake). Isobars: Cp from −1.5 to 0.75, dashed where suction.</div>
      {footer}
    </section>
  );
}
