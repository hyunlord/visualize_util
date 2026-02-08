import { memo } from 'react';
import type { FeatureInfo } from '../api/types';

interface FeatureCardProps {
  feature: FeatureInfo;
  onClick: () => void;
}

function FeatureCardComponent({ feature, onClick }: FeatureCardProps) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left p-4 rounded-xl border border-slate-700/50 bg-slate-800/50
                 hover:bg-slate-800 hover:border-slate-600 transition-all duration-200
                 group cursor-pointer"
    >
      <div className="flex items-start gap-3">
        {/* Color indicator */}
        <div
          className="w-1.5 h-full min-h-[60px] rounded-full flex-shrink-0 mt-1"
          style={{ backgroundColor: feature.color }}
        />

        <div className="flex-1 min-w-0">
          {/* Feature name */}
          <h3 className="text-base font-semibold text-slate-100 group-hover:text-white truncate">
            {feature.name}
          </h3>

          {/* Description */}
          {feature.description && (
            <p className="text-sm text-slate-400 mt-1 line-clamp-2">
              {feature.description}
            </p>
          )}

          {/* Flow summary */}
          {feature.flow_summary && (
            <p className="text-xs text-slate-500 mt-1.5 line-clamp-2 italic">
              {feature.flow_summary}
            </p>
          )}

          {/* Stats */}
          <div className="flex items-center gap-3 mt-3 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3z" />
              </svg>
              {feature.node_count} nodes
            </span>
            <span className="text-slate-600">|</span>
            <span className="flex items-center gap-1 text-slate-500 group-hover:text-blue-400 transition-colors">
              View Flow
              <svg className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </span>
          </div>
        </div>
      </div>
    </button>
  );
}

export default memo(FeatureCardComponent);
