// The pages (tabs) of the site, chosen by the #hash of the address.

export const PAGES = ['home', 'explore', 'draw', 'pinn', 'results'];
export const NAV_TITLES = { home: 'Home', explore: 'Explore', draw: 'Draw', pinn: 'PINN Lab', results: 'Results' };

/** "#explore" -> "explore"; null for anything that is not a page (for example "#credit" or an empty hash). */
export function pageFromHash(hash) {
  const h = (hash ?? '').replace(/^#/, '');
  return PAGES.includes(h) ? h : null;
}
