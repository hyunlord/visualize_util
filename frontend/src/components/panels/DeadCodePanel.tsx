import { useGraphStore } from '../../stores/graphStore';

export default function DeadCodePanel({ onClose }: { onClose: () => void }) {
  const { deadCode, selectNode } = useGraphStore();

  if (deadCode.length === 0) {
    return (
      <div className="absolute left-0 top-0 h-full w-80 bg-slate-900 border-r border-slate-700 z-10 p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-slate-100">Dead Code</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200">&times;</button>
        </div>
        <p className="text-slate-400 text-sm">No dead code detected.</p>
      </div>
    );
  }

  return (
    <div className="absolute left-0 top-0 h-full w-80 bg-slate-900 border-r border-slate-700 z-10 overflow-y-auto">
      <div className="p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-slate-100">
            Dead Code ({deadCode.length})
          </h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200">&times;</button>
        </div>

        <div className="space-y-3">
          {deadCode.map((item) => (
            <button
              key={item.node_id}
              onClick={() => selectNode(item.node_id)}
              className="w-full text-left p-3 rounded-lg bg-slate-800/50 border border-red-500/20 hover:border-red-500/40 transition-colors"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">
                  {item.node_type}
                </span>
                <span className="text-xs text-slate-500">
                  {Math.round(item.confidence * 100)}% confident
                </span>
              </div>
              <div className="text-sm font-mono text-slate-200">{item.name}</div>
              <div className="text-xs text-slate-500 mt-1">{item.file_path}:{item.line_start}</div>
              <div className="text-xs text-slate-400 mt-1">{item.reason}</div>
              {item.llm_explanation && (
                <div className="text-xs text-amber-400/80 mt-1 italic">
                  {item.llm_explanation}
                </div>
              )}
              {item.suggested_feature && (
                <div className="text-xs text-blue-400 mt-1">
                  Suggested: {item.suggested_feature}
                </div>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
