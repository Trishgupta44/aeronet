import { useEffect, useState } from 'react';
import { getCapabilities } from '../api.js';

const OPEN = { fvm_sample: true, fvm_curves: true, public_build: false, loaded: false };

/**
 * What this deployment offers (GET /api/capabilities, also saved in the static demo's bundle): `{fvm_sample, fvm_curves, public_build, loaded}`.
 * Until the answer arrives, or when it cannot be had, nothing is reported as missing (the page then behaves as the full app).
 */
export function useCapabilities() {
  const [caps, setCaps] = useState(OPEN);
  useEffect(() => {
    let live = true;
    getCapabilities()
      .then((c) => live && setCaps({ ...OPEN, ...c, loaded: true }))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  return caps;
}
