import { useEffect, useState } from 'react';
import { solve, solveShape } from '../api.js';

const FAST_GRID = 100; // low-resolution field while a slider is being dragged
const FULL_GRID = 170; // full-resolution field once the input settles
const SETTLE_MS = 180;

/**
 * Ask the backend for a solution whenever the inputs change.
 *
 * `input` is `{kind:'naca', solver, code, alpha, mach, reference}` or
 * `{kind:'shape', points, alpha, mach, lifting, ground, reference}`, or null (nothing to solve: the result is cleared).
 * A quick low-resolution request goes out immediately; if the inputs stay unchanged for SETTLE_MS, a full-resolution
 * request replaces it. Stale requests are aborted, and the previous result stays visible while a new one loads.
 * `points` is compared by identity: a new array means a new shape.
 *
 * Returns `{result, error, failedPoints}`; `failedPoints` is the `points` array of the shape the error belongs to,
 * so a caller can tell a rejected drawing from a transient failure.
 */
export function useSolve(input) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [failedPoints, setFailedPoints] = useState(null);

  const kind = input?.kind ?? null;
  const { solver, code, points, alpha, mach, lifting, ground, reference } = input ?? {};

  useEffect(() => {
    if (!kind) {
      setResult(null);
      setError(null);
      setFailedPoints(null);
      return undefined;
    }
    let cancelled = false;
    const controllers = [];
    const ref = reference?.points ?? null;
    const run = async (gridWidth) => {
      const controller = new AbortController();
      controllers.push(controller);
      try {
        const data =
          kind === 'shape'
            ? await solveShape({ points, alpha, mach, lifting, ground: ground ?? null, grid_width: gridWidth, reference: ref }, controller.signal)
            : await solve({ solver, code, alpha, mach, grid_width: gridWidth, reference: ref }, controller.signal);
        if (!cancelled) {
          setResult(data);
          setError(null);
          setFailedPoints(null);
        }
      } catch (e) {
        if (e.name !== 'AbortError' && !cancelled) {
          setError(e.message);
          setFailedPoints(points ?? null);
        }
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
  }, [kind, solver, code, points, alpha, mach, lifting, ground, reference]);

  return { result, error, failedPoints };
}
