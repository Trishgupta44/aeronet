// Thin client for the AeroNet backend (all paths are relative, so the same code works with the
// Vite dev proxy and when FastAPI serves the built frontend).

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    throw new Error('Cannot reach the solver API.');
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === 'string') detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch {
      /* keep the generic message */
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
export const getViscous = (body, signal) => request('/api/viscous', { method: 'POST', body: JSON.stringify(body), signal });
export const solve = (body, signal) => request('/api/solve', { method: 'POST', body: JSON.stringify(body), signal });
export const solveShape = (body, signal) => request('/api/solve_shape', { method: 'POST', body: JSON.stringify(body), signal });
export const getShapes = () => request('/api/shapes');
export const parseDat = (text, name) => request('/api/shapes/parse_dat', { method: 'POST', body: JSON.stringify({ text, name }) });
export const getPolar = (body, signal) => request('/api/polar', { method: 'POST', body: JSON.stringify(body), signal });
