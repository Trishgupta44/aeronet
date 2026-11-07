// 2x2 grid of link cards: thumbnail slot, uppercase compressed title, "Find out more" arrow link.
export default function CardGrid({ items }) {
  return (
    <ul className="cardgrid">
      {items.map((it) => (
        <li key={it.title}>
          <a className="tile" href={it.href}>
            <span className="tile-thumb" aria-hidden="true">{it.thumb}</span>
            <span className="tile-body">
              <span className="tile-title">{it.title}</span>
              <span className="tile-text">{it.text}</span>
              <span className="arrow-link">Find out more <span aria-hidden="true">→</span></span>
            </span>
          </a>
        </li>
      ))}
    </ul>
  );
}
