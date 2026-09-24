import { useEffect, useState } from 'react';

/** Redraw counter that changes when the OS light/dark preference changes (canvas colours come from CSS variables). */
function useThemeTick() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!mq) return undefined;
    const on = () => setTick((t) => t + 1);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return tick;
}

/**
 * Keep a <canvas> sharp at any width: sizes it to its parent (with devicePixelRatio) and calls
 * draw(ctx, width, height) whenever the size or `deps` change.
 */
export function useCanvasDraw(canvasRef, getHeight, draw, deps) {
  const [width, setWidth] = useState(0);
  const themeTick = useThemeTick();

  useEffect(() => {
    const el = canvasRef.current?.parentElement;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, [canvasRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;
    const height = getHeight(width);
    const dpr = window.devicePixelRatio || 1;
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, width, height);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, themeTick, ...deps]);
}

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
