import { useEffect, useSyncExternalStore } from 'react';
import { getSnapshot, retry, startHealthMonitor, subscribe } from '../offline/offline.js';

const when = (iso) => {
  if (!iso) return 'an unknown date';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
};

/** Shown only when the backend is unreachable: says that the page now shows saved results, from when, and what still works. */
export default function BackendBanner() {
  const s = useSyncExternalStore(subscribe, getSnapshot);
  useEffect(() => startHealthMonitor(), []);
  if (!s.offline) return null;
  return (
    <div className="notice error-banner" role="alert" style={{ margin: 0, borderRadius: 0 }}>
      <b>The solver backend is not reachable.</b>{' '}
      {s.hasBundle ? (
        <>
          Showing saved results from <b>{when(s.savedAt)}</b>. The presets at their starting angle and Mach number, the PINN results and the charts work; the
          sliders, drawing and new cases need the backend.
        </>
      ) : (
        <>No saved results are available either, so nothing can be shown until the backend is back.</>
      )}{' '}
      <button type="button" className="btn" onClick={retry}>Try again</button>
      {s.retryFailed && <> Still not reachable.</>}
    </div>
  );
}
