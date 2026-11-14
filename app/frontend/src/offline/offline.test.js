import { afterEach, describe, expect, it, vi } from 'vitest';
import { getAirfoils, solve } from '../api.js';
import { cacheKey, hashPoints } from './cacheKey.js';
import { __setForTest, getSnapshot, isOffline, offlineLookup } from './offline.js';

const square = [[0, 0], [1, 0], [1, 1], [0, 1]];

describe('cacheKey', () => {
  it('ignores the grid width and the contents of a reference, but not the inputs that change the answer', () => {
    const a = { solver: 'panel', code: '0012', alpha: 4, mach: 0.4, grid_width: 100, reference: null };
    expect(cacheKey('/api/solve', { ...a, grid_width: 170 })).toBe(cacheKey('/api/solve', a));
    expect(cacheKey('/api/solve', { ...a, alpha: 5 })).not.toBe(cacheKey('/api/solve', a));
    expect(cacheKey('/api/solve', { ...a, reference: [[0, 0, 1]] })).not.toBe(cacheKey('/api/solve', a));
    expect(cacheKey('/api/solve', { ...a, reference: [[0, 0, 1]] })).toBe(cacheKey('/api/solve', { ...a, reference: [[9, 9, 9]] }));
  });
  it('hashes an outline by its points', () => {
    expect(hashPoints(square)).toBe(hashPoints(square.map((p) => [...p])));
    expect(hashPoints(square)).not.toBe(hashPoints([[0, 0], [1, 0], [1, 1], [0, 1.01]]));
    const s = (extra) => cacheKey('/api/solve_shape', { points: square, alpha: 4, mach: 0, lifting: true, ground: null, grid_width: 100, reference: null, ...extra });
    expect(s({ grid_width: 170 })).toBe(s({}));
    expect(s({ ground: 0.1 })).not.toBe(s({}));
  });
});

describe('offline fallback', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    __setForTest({ bundle: null, offline: false });
  });

  it('serves a saved answer and a clear error for a case that was not saved', async () => {
    const body = { solver: 'panel', code: '0012', alpha: 4, mach: 0.4, grid_width: 100, reference: null };
    __setForTest({ bundle: { generated_utc: '2026-11-08T10:00:00Z', entries: { [cacheKey('/api/solve', body)]: { cl: 0.5 } } }, offline: true });
    expect(await offlineLookup('/api/solve', body)).toEqual({ cl: 0.5 });
    await expect(offlineLookup('/api/solve', { ...body, alpha: 7 })).rejects.toThrow(/^Cannot reach the solver API, and this case was not saved/);
  });

  it('switches to the saved answers when the network fails, and stops calling the network afterwards', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    vi.stubGlobal('fetch', fetchMock);
    __setForTest({ bundle: { generated_utc: '2026-11-08T10:00:00Z', entries: { '/api/airfoils': [{ code: '0012', label: 'NACA 0012' }] } }, offline: false });
    expect(await getAirfoils()).toEqual([{ code: '0012', label: 'NACA 0012' }]);
    expect(isOffline()).toBe(true);
    expect(getSnapshot()).toMatchObject({ offline: true, savedAt: '2026-11-08T10:00:00Z', hasBundle: true });
    const calls = fetchMock.mock.calls.length;
    await getAirfoils();
    await expect(solve({ solver: 'panel', code: '9999', alpha: 1, mach: 0.1 })).rejects.toThrow(/^Cannot reach/);
    expect(fetchMock.mock.calls.length).toBe(calls); // no further requests while offline
  });

  it('treats a 5xx without a JSON detail (a dev-proxy error) as unreachable, but passes real API errors on', async () => {
    __setForTest({ bundle: { generated_utc: 'x', entries: { '/api/airfoils': [1] } }, offline: false });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => { throw new Error('not json'); } }));
    expect(await getAirfoils()).toEqual([1]);
    expect(isOffline()).toBe(true);

    __setForTest({ bundle: { generated_utc: 'x', entries: {} }, offline: false });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: 'PINN results are not available here' }) }));
    await expect(getAirfoils()).rejects.toThrow('PINN results are not available here');
    expect(isOffline()).toBe(false);
  });
});
