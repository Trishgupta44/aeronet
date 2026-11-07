// Decorative thumbnails for the cards band. They are drawings, not data.
const base = { viewBox: '0 0 320 150', preserveAspectRatio: 'xMidYMid slice', focusable: 'false' };

function Flow({ children }) {
  return (
    <svg {...base}>
      <rect width="320" height="150" fill="#0a1a5c" />
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
    <rect width="320" height="150" fill="#dde7ee" />
    <path d="M70 90 C 100 40, 190 36, 250 84 C 190 72, 120 86, 70 90 Z" fill="none" stroke="#1a0f9e" strokeWidth="2.4" strokeDasharray="6 5" strokeLinecap="round" />
    <path d="M236 40 l22 22 l-6 2 l-2 6 z" fill="#1e3137" />
  </svg>
);

export const LabThumb = () => (
  <svg {...base}>
    <rect width="320" height="150" fill="#1e3137" />
    {[0, 1, 2, 3].map((r) =>
      [0, 1, 2, 3, 4, 5, 6, 7].map((c) => <circle key={`${r}-${c}`} cx={50 + c * 32} cy={34 + r * 28} r="3" fill="#fff" fillOpacity=".45" />),
    )}
  </svg>
);

export const ResultsThumb = () => (
  <svg {...base}>
    <rect width="320" height="150" fill="#dde7ee" />
    {[34, 60, 86, 112].map((y) => (
      <g key={y}>
        <line x1="40" x2="280" y1={y} y2={y} stroke="#1a0f9e" strokeOpacity=".35" />
        <rect x="40" y={y - 14} width="70" height="6" fill="#1a0f9e" fillOpacity=".7" />
        <rect x="130" y={y - 14} width="40" height="6" fill="#1e3137" fillOpacity=".5" />
      </g>
    ))}
  </svg>
);
