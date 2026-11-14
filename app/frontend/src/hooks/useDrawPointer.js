import { useEffect, useRef } from 'react';

// The drawing window is the field window of the solver (x from -0.4 to 1.4, y from -0.5 to 0.5; see backend field.py).
export const WINDOW = { xmin: -0.4, xmax: 1.4, ymin: -0.5, ymax: 0.5 };

/** Pixel position inside the canvas rectangle -> world point, using the same window mapping as the flow canvas. */
export function toWorldPoint(rect, clientX, clientY) {
  const fx = (clientX - rect.left) / rect.width;
  const fy = (clientY - rect.top) / rect.height;
  return [WINDOW.xmin + fx * (WINDOW.xmax - WINDOW.xmin), WINDOW.ymax - fy * (WINDOW.ymax - WINDOW.ymin)];
}

/**
 * Turn pointer events on a canvas into a stroke of world points.
 * A mouse or pen always draws; a finger draws only while the pen is `armed` (otherwise the page scrolls as usual).
 * `onLive(points)` fires while drawing (points is the stroke so far), `onComplete(points)` on pointer-up and
 * `onLive(null)` after any end of the stroke. Uses pointer capture, so leaving the canvas does not end the stroke.
 */
export function useDrawPointer(canvasRef, armed, { onLive, onComplete }) {
  const cb = useRef({ onLive, onComplete });
  cb.current = { onLive, onComplete };
  const armedRef = useRef(armed);
  armedRef.current = armed;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    let stroke = null;
    let id = null;

    const point = (e) => toWorldPoint(canvas.getBoundingClientRect(), e.clientX, e.clientY);
    const down = (e) => {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      if (e.pointerType === 'touch' && !armedRef.current) return;
      stroke = [point(e)];
      id = e.pointerId;
      try {
        canvas.setPointerCapture?.(id);
      } catch {
        /* a synthetic pointer cannot be captured; drawing still works while the pointer stays over the canvas */
      }
      cb.current.onLive?.(stroke);
      e.preventDefault();
    };
    const move = (e) => {
      if (!stroke || e.pointerId !== id) return;
      const events = e.getCoalescedEvents?.() ?? [];
      for (const ev of events.length ? events : [e]) stroke.push(point(ev));
      cb.current.onLive?.(stroke);
    };
    const end = (e, ok) => {
      if (!stroke || e.pointerId !== id) return;
      const done = stroke;
      stroke = null;
      id = null;
      try {
        canvas.releasePointerCapture?.(e.pointerId);
      } catch {
        /* not captured */
      }
      cb.current.onLive?.(null);
      if (ok) cb.current.onComplete?.(done);
    };
    const up = (e) => end(e, true);
    const cancel = (e) => end(e, false);

    canvas.addEventListener('pointerdown', down);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', cancel);
    return () => {
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointercancel', cancel);
    };
  }, [canvasRef]);
}
