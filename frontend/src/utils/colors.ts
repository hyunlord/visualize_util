const FEATURE_COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#ef4444', // red
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
  '#6366f1', // indigo
];

export function getFeatureColor(index: number): string {
  return FEATURE_COLORS[index % FEATURE_COLORS.length];
}

export function getNodeBorderColor(featureColor: string | null, isDeadCode: boolean, isEntryPoint: boolean): string {
  if (isDeadCode) return '#ef4444';
  if (isEntryPoint) return '#f59e0b';
  return featureColor || '#475569';
}

export function getNodeBgColor(featureColor: string | null, isDeadCode: boolean): string {
  if (isDeadCode) return '#1e1215';
  if (featureColor) return featureColor + '15';
  return '#1e293b';
}

export function getEdgeColor(isLlmInferred: boolean, edgeType: string): string {
  if (isLlmInferred) return '#f59e0b';
  if (edgeType === 'imports') return '#475569';
  if (edgeType === 'inherits') return '#8b5cf6';
  return '#64748b';
}
