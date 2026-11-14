// The pages (tabs) of the site, chosen by the #hash of the address.

export const PAGES = ['home', 'explore', 'draw', 'pinn', 'results'];
export const NAV_TITLES = { home: 'Home', explore: 'Explore', draw: 'Draw', pinn: 'PINN Lab', results: 'Results' };

/** "#explore" -> "explore"; null for anything that is not a page (for example "#credit" or an empty hash). */
export function pageFromHash(hash) {
  const h = (hash ?? '').replace(/^#/, '');
  return PAGES.includes(h) ? h : null;
}

/**
 * The page an address opens: the #hash page if it names one; otherwise a PINN Lab deep link (?pinn=<case>&round=N, written before the
 * pages existed) opens the Lab, and a tool link (?tab=explore|draw) opens that tool; anything else is the home page.
 */
export function pageFromAddress(hash, search) {
  const fromHash = pageFromHash(hash);
  if (fromHash) return fromHash;
  const q = new URLSearchParams(search ?? '');
  if (q.has('pinn')) return 'pinn';
  const tab = q.get('tab');
  return tab === 'explore' || tab === 'draw' ? tab : 'home';
}
