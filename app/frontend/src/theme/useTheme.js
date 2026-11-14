import { useSyncExternalStore } from 'react';
import { getSnapshot, setMode, subscribe } from './theme.js';

/** `{mode, resolved, setMode}`: mode is the choice (system, light, dark), resolved is what is on screen (light or dark). */
export function useTheme() {
  const s = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return { mode: s.mode, resolved: s.resolved, setMode };
}
