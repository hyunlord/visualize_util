import { useGraphStore } from '../../stores/graphStore';

export default function FlowNodeDetail() {
  const { featureFlow, selectedNodeId, selectNode } = useGraphStore();

  if (!featureFlow || !selectedNodeId) return null;

  const step = featureFlow.flow_steps.find((s) => s.node_id === selectedNodeId);
  if (!step) return null;

  return (
    <>
      {/* Backdrop overlay */}
      <div
        className="fixed inset-0 bg-black/30 z-40"
        onClick={() => selectNode(null)}
      />

      {/* Detail panel */}
      <div className="absolute top-0 right-0 w-[28rem] h-full bg-slate-900 border-l border-slate-800
                      shadow-2xl z-50 overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 px-5 py-3.5 flex items-center justify-between">
          <h3 className="text-base font-semibold text-slate-100 truncate font-mono">
            {step.function}
          </h3>
          <button
            onClick={() => selectNode(null)}
            className="text-slate-500 hover:text-slate-300 p-1.5 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-5">
          {/* Step info */}
          <div>
            <span className="text-xs uppercase tracking-wider text-slate-500 font-medium">
              Step {step.order}
            </span>
            <p className="text-sm text-slate-300 mt-1.5 leading-relaxed">
              {step.description}
            </p>
          </div>

          {/* File info */}
          <div>
            <span className="text-xs uppercase tracking-wider text-slate-500 font-medium">
              Location
            </span>
            <p className="text-sm font-mono text-slate-400 mt-1.5">
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
              <span className="text-xs uppercase tracking-wider text-slate-500 font-medium">
                Source Code
              </span>
              <pre className="mt-1.5 p-4 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto max-h-[500px] overflow-y-auto leading-relaxed">
                {step.source_code}
              </pre>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
