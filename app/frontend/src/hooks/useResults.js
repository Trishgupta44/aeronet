import { useEffect, useState } from 'react';
import { getResults } from '../api.js';

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
