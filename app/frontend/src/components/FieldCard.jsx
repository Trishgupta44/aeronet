import { useRef } from 'react';
import { cmap, gradientCss } from '../utils/colormap.js';
import { useCanvasDraw } from '../hooks/useCanvasDraw.js';

const offscreen = typeof document !== 'undefined' ? document.createElement('canvas') : null;

function drawField(g, w, h, result) {
  g.fillStyle = '#f4f3ee';
  g.fillRect(0, 0, w, h);
  const f = result?.field;
  if (!f) return;

  // colour the grid on an off-screen canvas, then scale it up smoothly
  offscreen.width = f.width;
  offscreen.height = f.height;
  const og = offscreen.getContext('2d');
  const img = og.createImageData(f.width, f.height);
  for (let k = 0; k < f.cp.length; k++) {
    const v = f.cp[k];
    if (v === null) continue; // inside the body: stays transparent
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

  const px = (x) => ((x - f.xmin) / (f.xmax - f.xmin)) * w;
  const py = (y) => ((f.ymax - y) / (f.ymax - f.ymin)) * h;
  g.beginPath();
  result.x.forEach((x, i) => {
    const X = px(x);
    const Y = py(result.y[i]);
    if (i) g.lineTo(X, Y);
    else g.moveTo(X, Y);
  });
  g.closePath();
  g.fillStyle = '#20242c';
  g.fill();
  g.strokeStyle = '#0c0e12';
  g.lineWidth = 1.2;
  g.stroke();

  // freestream arrow, tilted by the angle of attack
  const a = (result.alpha * Math.PI) / 180;
  const ax = 34;
  const ay = h / 2;
  const L = 44;
  const ex = ax + L * Math.cos(a);
  const ey = ay - L * Math.sin(a);
  g.strokeStyle = '#20242c';
  g.fillStyle = '#20242c';
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(ax, ay);
  g.lineTo(ex, ey);
  g.stroke();
  g.beginPath();
  g.moveTo(ex, ey);
  g.lineTo(ex - 9 * Math.cos(a - 0.45), ey + 9 * Math.sin(a - 0.45));
  g.lineTo(ex - 9 * Math.cos(a + 0.45), ey + 9 * Math.sin(a + 0.45));
  g.closePath();
  g.fill();
  g.font = '12px system-ui';
  g.fillText('freestream', ax - 8, ay + 30);
}

export default function FieldCard({ result }) {
  const ref = useRef(null);
  useCanvasDraw(ref, (w) => Math.round(w / 1.8), (g, w, h) => drawField(g, w, h, result), [result]);
  return (
    <section className="card">
      <h2>Pressure field (Cp)</h2>
      <div className="canvas-box">
        <canvas id="field" ref={ref} aria-label="Pressure coefficient field around the airfoil" role="img" />
      </div>
      <div className="cbar">
        <span>−2 suction</span>
        <div className="bar" id="bar" style={{ background: gradientCss() }} />
        <span>+1 stagnation</span>
      </div>
    </section>
  );
}
