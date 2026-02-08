import type { FeatureInfo } from '../api/types';
import FeatureCard from '../components/FeatureCard';

interface FeatureOverviewProps {
  features: FeatureInfo[];
  onSelectFeature: (featureId: string) => void;
}

export default function FeatureOverview({ features, onSelectFeature }: FeatureOverviewProps) {
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
      </div>
    </div>
  );
}
