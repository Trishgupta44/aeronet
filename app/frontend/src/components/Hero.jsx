// Full-bleed hero. `children` is the background slot (B7 puts the live flow canvas there);
// without it a static dark gradient is shown.
export default function Hero({ children, caption }) {
  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="hero-bg" aria-hidden="true">{children ?? <div className="hero-placeholder" />}</div>
      <div className="hero-shade" aria-hidden="true" />
      <div className="container hero-copy">
        <p className="eyebrow eyebrow--light">An airfoil flow explorer</p>
        <h1 id="hero-title">See the air</h1>
        <p className="hero-tag">Physics-informed airfoil flow, on a laptop</p>
        <div className="hero-actions">
          <a className="button button--primary" href="#explore">Try the explorer</a>
          <a className="button button--outline" href="#draw">Draw your own shape</a>
        </div>
        {caption && <p className="hero-caption">{caption}</p>}
      </div>
    </section>
  );
}
