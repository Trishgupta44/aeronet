import { useEffect, useSyncExternalStore } from 'react';
import { getSnapshot, retry, startHealthMonitor, subscribe } from '../offline/offline.js';

const when = (iso) => {
  if (!iso) return 'an unknown date';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
};

/**
 * Shown only when the page runs on saved results: because the backend is unreachable (red, with Try again), or because this is a
 * static demo build (calm notice, nothing to retry). Says from when the results are and what still works.
 */
export default function BackendBanner() {
  const s = useSyncExternalStore(subscribe, getSnapshot);
  useEffect(() => startHealthMonitor(), []);
  if (!s.offline) return null;

  if (s.staticDemo) {
    return (
      <div className="notice" role="status" style={{ margin: 0, borderRadius: 0 }}>
        <b>Static demo.</b>{' '}
        {s.hasBundle ? (
          <>Every result on this page is a saved result from <b>{when(s.savedAt)}</b>: the presets at their starting angle and Mach number, the PINN results and the charts. The sliders, drawing and new cases need the live app.</>
        ) : (
          <>The saved results could not be loaded, so nothing can be shown.</>
        )}
      </div>
    );
  }

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
