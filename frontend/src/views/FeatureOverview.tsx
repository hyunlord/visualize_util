import { useState } from 'react';
import type { FeatureInfo, DeadCodeItem } from '../api/types';
import FeatureCard from '../components/FeatureCard';

interface FeatureOverviewProps {
  features: FeatureInfo[];
  deadCode?: DeadCodeItem[];
  onSelectFeature: (featureId: string) => void;
}

export default function FeatureOverview({ features, deadCode = [], onSelectFeature }: FeatureOverviewProps) {
  const [showDeadCode, setShowDeadCode] = useState(false);

  if (features.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-5xl mb-4 opacity-20">&#128269;</div>
          <h2 className="text-lg font-semibold text-slate-400 mb-2">
            No features discovered yet
          </h2>
          <p className="text-sm text-slate-500">
            Run an analysis to discover features in this codebase.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-slate-100">
            Discovered Features
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            {features.length} features found. Click a feature to view its execution flow.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {features.map((feature) => (
            <FeatureCard
              key={feature.id}
              feature={feature}
              onClick={() => onSelectFeature(feature.id)}
            />
          ))}
        </div>

        {/* Dead Code Section */}
        {deadCode.length > 0 && (
          <div className="mt-8">
            <button
              onClick={() => setShowDeadCode(!showDeadCode)}
              className="flex items-center gap-2 text-sm font-semibold text-slate-400 hover:text-slate-200 transition-colors"
            >
              <svg
                className={`w-4 h-4 transition-transform ${showDeadCode ? 'rotate-90' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              Dead Code ({deadCode.length} items)
            </button>

            {showDeadCode && (
              <div className="mt-3 space-y-2">
                {deadCode.map((item) => (
                  <div
                    key={item.node_id}
                    className="px-4 py-3 rounded-lg bg-slate-800/50 border border-slate-700/50"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 font-medium">
                        DEAD
                      </span>
                      <span className="text-sm font-mono text-slate-300">
                        {item.name}
                      </span>
                    </div>
                    <div className="text-xs text-slate-500 mt-1 font-mono">
                      {item.file_path}:{item.line_start}-{item.line_end}
                    </div>
                    {item.reason && (
                      <div className="text-xs text-slate-500 mt-1">
                        {item.reason}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
