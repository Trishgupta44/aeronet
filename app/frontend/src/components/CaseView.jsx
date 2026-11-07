import { useRef, useState } from 'react';
import { saveBlob, saveText } from '../utils/download.js';
import { surfaceCsv } from '../utils/outline.js';
import CpCard from './CpCard.jsx';
import FlowCanvas from './FlowCanvas.jsx';
import { ResultsCard } from './InfoCards.jsx';
import PolarCard from './PolarCard.jsx';

/**
 * The right-hand column of a tool section: flow picture (with optional pen layer as `children`), a status slot under it,
 * the results (with pin and exports), the surface pressure plot and the lift curve.
 * `caseName` labels a pinned result; `polarSource` is `{code, mach}` or `{points, lifting, ground, mach}` (or null).
 */
export default function CaseView({ idp = '', caseName, result, reference, layers, onLayersChange, paused, onPausedChange, children, status, polarSource }) {
  const [pinned, setPinned] = useState(null);
  const stageRef = useRef(null);

  const onPin = () => {
    if (!result) return;
    setPinned({
      label: `${caseName} · ${result.alpha.toFixed(1)}° · Ma ${result.mach.toFixed(2)}`,
      cl: result.cl,
      surface: result.surface,
      x: result.x,
      y: result.y,
    });
  };
  const onPng = () => stageRef.current?.toBlob((b) => b && saveBlob(b, 'aeronet-flow.png'));
  const onCsv = () => result && saveText(surfaceCsv(result), 'aeronet-cp.csv', 'text/csv');

  return (
    <div className="stack">
      <FlowCanvas idp={idp} result={result} layers={layers} onLayersChange={onLayersChange} paused={paused} onPausedChange={onPausedChange}
        footer={status} ghost={pinned} stageRef={stageRef}>
        {children}
      </FlowCanvas>
      <ResultsCard idp={idp} result={result} pinned={pinned} onPin={onPin} onUnpin={() => setPinned(null)} onPng={onPng} onCsv={onCsv} />
      <CpCard idp={idp} result={result} reference={reference} pinned={pinned} />
      <PolarCard idp={idp} source={polarSource} current={result ? { alpha: result.alpha, cl: result.cl } : null} />
    </div>
  );
}
