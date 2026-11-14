// Decorative thumbnails for the cards band. They are drawings, not data. Colours come from the --thumb-* tokens so they follow the theme.
const base = { viewBox: '0 0 320 150', preserveAspectRatio: 'xMidYMid slice', focusable: 'false' };
const fill = (v) => ({ fill: `var(${v})` });
const stroke = (v) => ({ stroke: `var(${v})` });

function Flow({ children }) {
  return (
    <svg {...base}>
      <rect width="320" height="150" style={fill('--thumb-flow-bg')} />
      {[40, 62, 84, 106].map((y, i) => (
        <path
          key={y}
          d={`M0 ${y + 10} C 90 ${y + 10 - 5 * (i + 1)}, 150 ${y - 12 - i * 3}, 320 ${y + 6}`}
          stroke="#fff"
          strokeOpacity={0.55 - i * 0.08}
          strokeWidth="1.4"
          fill="none"
        />
      ))}
      {children}
    </svg>
  );
}

export const ExploreThumb = () => (
  <Flow>
    <path d="M110 80 C 140 56, 190 56, 222 78 C 190 98, 144 96, 110 80 Z" fill="#e8ecf4" />
  </Flow>
);

export const DrawThumb = () => (
  <svg {...base}>
    <rect width="320" height="150" style={fill('--thumb-bg')} />
    <path d="M70 90 C 100 40, 190 36, 250 84 C 190 72, 120 86, 70 90 Z" fill="none" style={stroke('--thumb-ink')} strokeWidth="2.4" strokeDasharray="6 5" strokeLinecap="round" />
    <path d="M236 40 l22 22 l-6 2 l-2 6 z" style={fill('--thumb-dark')} />
  </svg>
);

export const LabThumb = () => (
  <svg {...base}>
    <rect width="320" height="150" style={fill('--thumb-lab')} />
    {[0, 1, 2, 3].map((r) =>
      [0, 1, 2, 3, 4, 5, 6, 7].map((c) => <circle key={`${r}-${c}`} cx={50 + c * 32} cy={34 + r * 28} r="3" fill="#fff" fillOpacity=".45" />),
    )}
  </svg>
);

export const ResultsThumb = () => (
  <svg {...base}>
    <rect width="320" height="150" style={fill('--thumb-bg')} />
    {[34, 60, 86, 112].map((y) => (
      <g key={y}>
        <line x1="40" x2="280" y1={y} y2={y} style={stroke('--thumb-ink')} strokeOpacity=".35" />
        <rect x="40" y={y - 14} width="70" height="6" style={fill('--thumb-ink')} fillOpacity=".7" />
        <rect x="130" y={y - 14} width="40" height="6" style={fill('--thumb-dark')} fillOpacity=".5" />
      </g>
    ))}
  </svg>
);
