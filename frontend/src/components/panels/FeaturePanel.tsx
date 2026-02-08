import { useGraphStore } from '../../stores/graphStore';

export default function FeaturePanel({ onClose }: { onClose: () => void }) {
  const { features, activeFeatureIds, toggleFeature, showAllFeatures, showOnlyFeature } =
    useGraphStore();

  return (
    <div className="absolute left-0 top-0 h-full w-72 bg-slate-900 border-r border-slate-700 z-10 overflow-y-auto">
      <div className="p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-slate-100">Features</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200">&times;</button>
        </div>

        <button
          onClick={showAllFeatures}
          className="w-full text-sm text-center py-1.5 rounded bg-slate-800 text-slate-300 hover:bg-slate-700 mb-3"
        >
          Show All
        </button>

        <div className="space-y-2">
          {features.map((feature) => {
            const isActive = activeFeatureIds.has(feature.id);
            return (
              <div
                key={feature.id}
                className="flex items-center gap-2 p-2 rounded-lg hover:bg-slate-800/50"
              >
                <button
                  onClick={() => toggleFeature(feature.id)}
                  className={`w-4 h-4 rounded border-2 flex-shrink-0 ${
                    isActive ? 'bg-current' : ''
                  }`}
                  style={{ borderColor: feature.color, color: feature.color }}
                />
                <button
                  onClick={() => showOnlyFeature(feature.id)}
                  className="flex-1 text-left min-w-0"
                >
                  <div className="text-sm text-slate-200 truncate">{feature.name}</div>
                  <div className="text-xs text-slate-500">
                    {feature.node_count} nodes
                    {!feature.auto_detected && ' (manual)'}
                  </div>
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
