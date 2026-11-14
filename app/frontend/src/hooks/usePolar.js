import { useEffect, useState } from 'react';
import { getPolar } from '../api.js';

export const POLAR_ALPHAS = Array.from({ length: 23 }, (_, k) => k - 10); // -10 .. 12 degrees

/**
 * Cl against angle of attack for the current airfoil code or drawn shape, at the current Mach number.
 * `input` is `{code}` or `{points, lifting, ground}` plus `mach`; it is requested 300 ms after the last change.
 * Does not depend on the angle of attack itself (the whole curve is computed), so dragging the slider costs nothing.
 */
export function usePolar(input) {
  const [polar, setPolar] = useState(null);
  const [error, setError] = useState(null);
  const { code, points, lifting, ground, mach } = input ?? {};
  const active = !!input && (code || points);

  useEffect(() => {
    if (!active) {
      setPolar(null);
      setError(null);
      return undefined;
    }
    const controller = new AbortController();
    const t = setTimeout(async () => {
      try {
        const body = code
          ? { code, alphas: POLAR_ALPHAS, mach }
          : { points, alphas: POLAR_ALPHAS, mach, lifting, ground: ground ?? null };
        const data = await getPolar(body, controller.signal);
        setPolar(data);
        setError(null);
      } catch (e) {
        if (e.name !== 'AbortError') setError(e.message);
      }
    }, 300);
    return () => {
      clearTimeout(t);
      controller.abort();
    };
  }, [active, code, points, lifting, ground, mach]);

  return { polar, error };
}
