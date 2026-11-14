// Our own mark: a cambered airfoil section with two flow lines. Drawn here, not copied from anywhere.
export default function Glyph({ size = 26 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false" className="glyph">
      <path d="M2 19 C 9 8, 21 8, 30 18 C 22 15, 11 16, 2 19 Z" fill="currentColor" />
      <path d="M2 25 H 22" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" opacity=".7" />
      <path d="M8 29 H 30" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" opacity=".45" />
    </svg>
  );
}
