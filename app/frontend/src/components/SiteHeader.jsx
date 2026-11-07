import { useEffect, useState } from 'react';
import { activeSection } from '../utils/scrollSpy.js';
import Glyph from './Glyph.jsx';

export const NAV = [
  { id: 'explore', label: 'Explore' },
  { id: 'draw', label: 'Draw' },
  { id: 'pinn', label: 'PINN Lab' },
  { id: 'results', label: 'Results' },
  { id: 'about', label: 'About' },
];

function useScrollState() {
  const [state, setState] = useState({ stuck: false, active: null });
  useEffect(() => {
    let raf = 0;
    const read = () => {
      raf = 0;
      const y = window.scrollY;
      const offsets = NAV.map((n) => {
        const el = document.getElementById(n.id);
        return el ? { id: n.id, top: el.getBoundingClientRect().top + y } : null;
      }).filter(Boolean);
      const next = { stuck: y > 64, active: activeSection(offsets, y) };
      setState((s) => (s.stuck === next.stuck && s.active === next.active ? s : next));
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(read);
    };
    read();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);
  return state;
}

function Links({ active, onPick }) {
  return NAV.map((n) => (
    <a key={n.id} href={`#${n.id}`} aria-current={active === n.id ? 'location' : undefined} onClick={onPick}>
      {n.label}
    </a>
  ));
}

export default function SiteHeader() {
  const { stuck, active } = useScrollState();
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
      <header className="site-header">
        <div className="utility">
          <div className="container utility-in">
            <span>Prototype · inviscid · subsonic · screening tool</span>
            <nav aria-label="Utility">
              <a href="#about">About</a>
              <a href="#credit">Credit</a>
            </nav>
          </div>
        </div>
        <div className="mainnav">
          <div className="container mainnav-in">
            <a className="brand" href="#top" aria-label="AeroNet, back to top">
              <Glyph /> <span>AeroNet</span>
            </a>
            <nav className="links" aria-label="Main">
              <Links active={active} />
            </nav>
            {burger('')}
          </div>
        </div>
      </header>

      <div className={`stickybar${stuck ? ' on' : ''}`} aria-hidden={!stuck} inert={stuck ? undefined : ''}>
        <div className="container stickybar-in">
          <a className="brand" href="#top">
            <Glyph size={22} /> <span>AeroNet</span>
          </a>
          <nav className="links" aria-label="Sections">
            <Links active={active} />
          </nav>
          <a className="totop" href="#top">Back to top <span aria-hidden="true">↑</span></a>
          {burger('burger--bar')}
        </div>
      </div>

      <div id="sheet" className={`sheet${open ? ' open' : ''}`} role="dialog" aria-modal="true" aria-label="Menu" hidden={!open}>
        <div className="container sheet-top">
          <span className="brand"><Glyph /> <span>AeroNet</span></span>
          <button type="button" className="sheet-close" aria-label="Close menu" onClick={() => setOpen(false)}>×</button>
        </div>
        <nav className="container sheet-links" aria-label="Menu">
          <Links active={active} onPick={() => setOpen(false)} />
        </nav>
      </div>
    </>
  );
}
