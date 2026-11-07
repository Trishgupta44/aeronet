import CpCard from './CpCard.jsx';
import FlowCanvas from './FlowCanvas.jsx';
import { ResultsCard } from './InfoCards.jsx';

/**
 * The right-hand column of a tool section: flow picture (with optional pen layer as `children`),
 * a status slot under it, the results and the surface pressure plot.
 */
export default function CaseView({ idp = '', result, reference, layers, onLayersChange, paused, onPausedChange, children, status }) {
  return (
    <div className="stack">
      <FlowCanvas idp={idp} result={result} layers={layers} onLayersChange={onLayersChange} paused={paused} onPausedChange={onPausedChange} footer={status}>
        {children}
      </FlowCanvas>
      <ResultsCard idp={idp} result={result} />
      <CpCard idp={idp} result={result} reference={reference} />
    </div>
  );
}
