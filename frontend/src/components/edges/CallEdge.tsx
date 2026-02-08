import { memo } from 'react';
import { BaseEdge, getSmoothStepPath, type EdgeProps } from '@xyflow/react';
import { getEdgeColor } from '../../utils/colors';

interface CallEdgeData {
  edge_type: string;
  is_llm_inferred: boolean;
}

function CallEdgeComponent({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps) {
  const edgeData = data as unknown as CallEdgeData;
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const color = getEdgeColor(edgeData?.is_llm_inferred || false, edgeData?.edge_type || 'calls');

  return (
    <BaseEdge
      path={edgePath}
      markerEnd={markerEnd}
      style={{
        stroke: color,
        strokeWidth: edgeData?.edge_type === 'calls' ? 1.5 : 1,
        strokeDasharray: edgeData?.is_llm_inferred ? '5 3' : undefined,
      }}
    />
  );
}

export default memo(CallEdgeComponent);
