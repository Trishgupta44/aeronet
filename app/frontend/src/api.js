// Thin client for the AeroNet backend (all paths are relative, so the same code works with the
// Vite dev proxy and when FastAPI serves the built frontend).

import { goOffline, isOffline, offlineLookup } from './offline/offline.js';

// When the backend does not answer (network error, or a 5xx from the dev proxy without a JSON "detail"), the page switches to
// saved results (see offline/offline.js). Real API errors (422, 503 with a message) are passed on unchanged.
async function request(path, options = {}) {
  const body = options.body ? JSON.parse(options.body) : undefined;
  if (isOffline()) return offlineLookup(path, body);
  let res;
  try {
    res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    goOffline();
    return offlineLookup(path, body);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    let hasDetail = false;
    try {
      const err = await res.json();
      if (typeof err.detail === 'string') {
        detail = err.detail;
        hasDetail = true;
      } else if (Array.isArray(err.detail) && err.detail[0]?.msg) {
        detail = err.detail[0].msg;
        hasDetail = true;
      }
    } catch {
      /* keep the generic message */
    }
    if (res.status >= 500 && !hasDetail) {
      goOffline();
      return offlineLookup(path, body);
    }
    throw new Error(detail);
  }
  return res.json();
}

export const getSolvers = () => request('/api/solvers');
export const getAirfoils = () => request('/api/airfoils');
export const getSampleReference = () => request('/api/reference/sample');
export const parseReference = (text, name) => request('/api/reference/parse', { method: 'POST', body: JSON.stringify({ text, name }) });
export const getShowdown = () => request('/api/showdown');
export const getTradeoff = () => request('/api/tradeoff');
export const optimizeShape = (body) => request('/api/optimize', { method: 'POST', body: JSON.stringify(body) });
export const getViscous = (body, signal) => request('/api/viscous', { method: 'POST', body: JSON.stringify(body), signal });
export const solve = (body, signal) => request('/api/solve', { method: 'POST', body: JSON.stringify(body), signal });
export const solveShape = (body, signal) => request('/api/solve_shape', { method: 'POST', body: JSON.stringify(body), signal });
export const getShapes = () => request('/api/shapes');
export const parseDat = (text, name) => request('/api/shapes/parse_dat', { method: 'POST', body: JSON.stringify({ text, name }) });
export const getPolar = (body, signal) => request('/api/polar', { method: 'POST', body: JSON.stringify(body), signal });
export const getPinnCases = () => request('/api/pinn/cases');
export const getPinnCase = (id) => request(`/api/pinn/cases/${encodeURIComponent(id)}`);
export const getResults = () => request('/api/results');
