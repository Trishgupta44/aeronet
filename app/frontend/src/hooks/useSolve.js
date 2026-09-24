import { useEffect, useState } from 'react';
import { solve } from '../api.js';

const FAST_GRID = 100; // low-resolution field while a slider is being dragged
const FULL_GRID = 170; // full-resolution field once the input settles
const SETTLE_MS = 180;

/**
 * Ask the backend for a solution whenever the inputs change.
 * A quick low-resolution request goes out immediately; if the inputs stay unchanged for
 * SETTLE_MS, a full-resolution request replaces it. Stale requests are aborted, and the previous
 * result stays visible while a new one loads (no flicker).
 */
export function useSolve({ solver, code, alpha, mach, reference }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const controllers = [];
    const run = async (gridWidth) => {
      const controller = new AbortController();
      controllers.push(controller);
      try {
        const data = await solve({ solver, code, alpha, mach, grid_width: gridWidth, reference: reference?.points ?? null }, controller.signal);
        if (!cancelled) {
          setResult(data);
          setError(null);
        }
      } catch (e) {
        if (e.name !== 'AbortError' && !cancelled) setError(e.message);
      }
    };
    const quick = setTimeout(() => run(FAST_GRID), 0);
    const full = setTimeout(() => {
      controllers.forEach((c) => c.abort());
      run(FULL_GRID);
    }, SETTLE_MS);
    return () => {
      cancelled = true;
      clearTimeout(quick);
      clearTimeout(full);
      controllers.forEach((c) => c.abort());
    };
  }, [solver, code, alpha, mach, reference]);

  return { result, error };
}
