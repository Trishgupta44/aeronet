import { useEffect, useState } from 'react';
import Glyph from './Glyph.jsx';
import ThemeToggle from './ThemeToggle.jsx';

// The pages (tabs) of the site. Each link is a plain #hash link; App follows the hash, so back, forward and shared links work.
export const NAV = [
  { id: 'home', label: 'Home' },
  { id: 'explore', label: 'Explore' },
  { id: 'draw', label: 'Draw' },
  { id: 'pinn', label: 'PINN Lab' },
  { id: 'results', label: 'Results' },
];

/** True once the page has scrolled past the header (shows the compact bar). */
function useStuck() {
  const [stuck, setStuck] = useState(false);
  useEffect(() => {
    const on = () => setStuck(window.scrollY > 64);
    on();
    window.addEventListener('scroll', on, { passive: true });
    return () => window.removeEventListener('scroll', on);
  }, []);
  return stuck;
}

function Links({ page, onPick }) {
  return NAV.map((n) => (
    <a key={n.id} href={`#${n.id}`} aria-current={page === n.id ? 'page' : undefined} onClick={onPick}>
      {n.label}
    </a>
  ));
}

export default function SiteHeader({ page }) {
  const stuck = useStuck();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === 'Escape' && setOpen(false);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const burger = (cls) => (
    <button className={`burger ${cls}`} type="button" aria-label="Open menu" aria-expanded={open} aria-controls="sheet" onClick={() => setOpen(true)}>
      <span /><span /><span />
    </button>
  );

  return (
    <>
      <header className="site-header" id="top">
        <div className="utility">
          <div className="container utility-in">
            <span>Prototype · inviscid · subsonic · screening tool</span>
            <nav aria-label="Utility">
              <a href="#credit">Credit</a>
            </nav>
          </div>
        </div>
        <div className="mainnav">
          <div className="container mainnav-in">
            <a className="brand" href="#home" aria-label="AeroNet, home">
              <Glyph /> <span>AeroNet</span>
            </a>
            <nav className="links" aria-label="Main">
              <Links page={page} />
            </nav>
            <div className="nav-tools">
              <ThemeToggle />
              {burger('')}
            </div>
          </div>
        </div>
      </header>

      <div className={`stickybar${stuck ? ' on' : ''}`} aria-hidden={!stuck} inert={stuck ? undefined : ''}>
        <div className="container stickybar-in">
          <a className="brand" href="#home">
            <Glyph size={22} /> <span>AeroNet</span>
          </a>
          <nav className="links" aria-label="Pages">
            <Links page={page} />
          </nav>
          <ThemeToggle onChrome />
          {burger('burger--bar')}
        </div>
      </div>

      <div id="sheet" className={`sheet${open ? ' open' : ''}`} role="dialog" aria-modal="true" aria-label="Menu" hidden={!open}>
        <div className="container sheet-top">
          <span className="brand"><Glyph /> <span>AeroNet</span></span>
          <button type="button" className="sheet-close" aria-label="Close menu" onClick={() => setOpen(false)}>×</button>
        </div>
        <nav className="container sheet-links" aria-label="Menu">
          <Links page={page} onPick={() => setOpen(false)} />
        </nav>
        <div className="container sheet-theme"><ThemeToggle onChrome /></div>
      </div>
    </>
  );
}
