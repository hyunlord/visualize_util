import { useGraphStore } from '../../stores/graphStore';

export default function FlowNodeDetail() {
  const { featureFlow, selectedNodeId, selectNode } = useGraphStore();

  if (!featureFlow || !selectedNodeId) return null;

  const step = featureFlow.flow_steps.find((s) => s.node_id === selectedNodeId);
  if (!step) return null;

  return (
    <div className="absolute top-0 right-0 w-96 h-full bg-slate-900 border-l border-slate-800
                    shadow-2xl z-30 overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 px-4 py-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-100 truncate">
          {step.function}
        </h3>
        <button
          onClick={() => selectNode(null)}
          className="text-slate-500 hover:text-slate-300 p-1"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* Step info */}
        <div>
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            Step {step.order}
          </span>
          <p className="text-sm text-slate-300 mt-1">
            {step.description}
          </p>
        </div>

        {/* File info */}
        <div>
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            Location
          </span>
          <p className="text-xs font-mono text-slate-400 mt-1">
            {step.file}
            {step.line_start && (
              <span className="text-slate-600">
                :{step.line_start}
                {step.line_end && step.line_end !== step.line_start ? `-${step.line_end}` : ''}
              </span>
            )}
          </p>
        </div>

        {/* Source code */}
        {step.source_code && (
          <div>
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
              Source Code
            </span>
            <pre className="mt-1 p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto max-h-[400px] overflow-y-auto">
              {step.source_code}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
