// The source label of a number: measured (produced by code in this repo), quoted (copied from a paper) or fitted (tuned to data).
export default function Tag({ kind, children }) {
  const k = ['measured', 'quoted', 'fitted', 'computed'].includes(kind) ? kind : 'plain';
  return <span className={`tag tag--${k}`}>{children ?? kind}</span>;
}
