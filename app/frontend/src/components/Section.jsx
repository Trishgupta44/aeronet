// One idea per band: mono eyebrow, compressed uppercase title, content. `band` uses the pale background.
export default function Section({ id, eyebrow, title, band = false, aside, children }) {
  return (
    <section id={id} className={`section${band ? ' section--band' : ''}`} aria-labelledby={`${id}-title`}>
      <div className="container">
        <div className="section-head">
          <div>
            {eyebrow && <p className="eyebrow">{eyebrow}</p>}
            <h2 id={`${id}-title`} className="display">{title}</h2>
          </div>
          {aside && <div className="section-aside">{aside}</div>}
        </div>
        {children}
      </div>
    </section>
  );
}
