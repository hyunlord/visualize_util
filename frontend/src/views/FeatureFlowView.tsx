import { useMemo, useCallback, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeMouseHandler,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';

import FlowStepNode from '../components/nodes/FlowStepNode';
import type { FeatureFlowResponse, FlowStep } from '../api/types';

const nodeTypes = {
  flowStepNode: FlowStepNode,
};

const FLOW_NODE_WIDTH = 380;
const FLOW_NODE_HEIGHT = 150;

function layoutFlowNodes(steps: FlowStep[], edges: Edge[], featureColor: string): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: 'TB',
    nodesep: 120,
    ranksep: 160,
    marginx: 40,
    marginy: 40,
  });

  steps.forEach((step) => {
    if (step.node_id) {
      g.setNode(step.node_id, { width: FLOW_NODE_WIDTH, height: FLOW_NODE_HEIGHT });
    }
  });

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  return steps
    .filter((step) => step.node_id)
    .map((step) => {
      const pos = g.node(step.node_id!);
      return {
        id: step.node_id!,
        type: 'flowStepNode',
        position: {
          x: (pos?.x || 0) - FLOW_NODE_WIDTH / 2,
          y: (pos?.y || 0) - FLOW_NODE_HEIGHT / 2,
        },
        data: {
          label: step.function,
          file_path: step.file,
          description: step.description,
          order: step.order,
          is_entry_point: step.order === 1,
          source_code: step.source_code,
          feature_color: featureColor,
          line_start: step.line_start,
          line_end: step.line_end,
        },
      };
    });
}

interface FeatureFlowViewProps {
  flowData: FeatureFlowResponse;
  onBack: () => void;
  onNodeClick: (nodeId: string) => void;
}

export default function FeatureFlowView({ flowData, onBack, onNodeClick }: FeatureFlowViewProps) {
  const featureColor = flowData.feature.color;

  const flowEdges: Edge[] = useMemo(
    () =>
      flowData.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        type: 'smoothstep',
        animated: true,
        style: { stroke: featureColor, strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 14,
          height: 14,
          color: featureColor,
        },
      })),
    [flowData.edges, featureColor]
  );

  const flowNodes: Node[] = useMemo(
    () => layoutFlowNodes(flowData.flow_steps, flowEdges, featureColor),
    [flowData.flow_steps, flowEdges, featureColor]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  const handleNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      onNodeClick(node.id);
    },
    [onNodeClick]
  );

  return (
    <div className="h-full flex flex-col">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-4 py-3 bg-slate-900/80 border-b border-slate-800 flex-shrink-0">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg
                     bg-slate-800 border border-slate-700 text-slate-300
                     hover:bg-slate-700 hover:text-white transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Features
        </button>

        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: featureColor }}
          />
          <h2 className="text-sm font-semibold text-slate-100">
            {flowData.feature.name}
          </h2>
        </div>

        {flowData.flow_summary && (
          <p className="text-xs text-slate-500 truncate ml-2 flex-1">
            {flowData.flow_summary}
          </p>
        )}

        <span className="text-xs text-slate-600 flex-shrink-0">
          {flowData.flow_steps.length} steps
        </span>
      </div>

      {/* Flow graph */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          minZoom={0.3}
          maxZoom={1.5}
          defaultEdgeOptions={{ animated: true }}
        >
          <Background color="#1e293b" gap={24} size={1} />
          <Controls className="!bg-slate-800 !border-slate-700 !shadow-lg" />
          <MiniMap
            className="!bg-slate-800 !border-slate-700"
            nodeColor={() => featureColor}
            maskColor="rgba(0, 0, 0, 0.7)"
          />
        </ReactFlow>
      </div>
    </div>
  );
}
