import { useEffect, useState } from 'react';
import { getResults, getRoundsE1 } from '../api.js';

/** The rows of results/summary.csv (GET /api/results), loaded once: `{rows, error}`; `rows` is null until they arrive. */
export function useResults() {
  const [state, setState] = useState({ rows: null, error: '' });
  useEffect(() => {
    let live = true;
    getResults()
      .then((rows) => live && setState({ rows, error: '' }))
      .catch((e) => live && setState({ rows: null, error: e.message }));
    return () => {
      live = false;
    };
  }, []);
  return state;
}

/** The round-by-round data of the six E1 runs (GET /api/results/e1_rounds): `{data, error}`; `data` is null until it arrives or when it is missing. */
export function useRoundsE1() {
  const [state, setState] = useState({ data: null, error: '' });
  useEffect(() => {
    let live = true;
    getRoundsE1()
      .then((data) => live && setState({ data, error: '' }))
      .catch((e) => live && setState({ data: null, error: e.message }));
    return () => {
      live = false;
    };
  }, []);
  return state;
}
