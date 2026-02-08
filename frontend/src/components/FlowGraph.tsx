import { useCallback, useEffect, useMemo } from 'react';
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

import FunctionNode from './nodes/FunctionNode';
import CallEdge from './edges/CallEdge';
import { useGraphStore } from '../stores/graphStore';
import { applyDagreLayout } from '../utils/layout';

const nodeTypes = {
  functionNode: FunctionNode,
};

const edgeTypes = {
  callEdge: CallEdge,
};

export default function FlowGraph() {
  const { getFilteredNodes, getFilteredEdges, selectNode } = useGraphStore();

  const filteredNodes = useGraphStore(getFilteredNodes);
  const filteredEdges = useGraphStore(getFilteredEdges);

  const layoutedNodes = useMemo(
    () => applyDagreLayout(filteredNodes, filteredEdges),
    [filteredNodes, filteredEdges]
  );

  const flowEdges: Edge[] = useMemo(
    () =>
      filteredEdges.map((e) => ({
        ...e,
        type: 'callEdge',
        markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12 },
      })),
    [filteredEdges]
  );

  const flowNodes: Node[] = useMemo(
    () =>
      layoutedNodes.map((n) => ({
        ...n,
        type: 'functionNode',
      })),
    [layoutedNodes]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  const onNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      onPaneClick={onPaneClick}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      fitView
      minZoom={0.1}
      maxZoom={2}
      defaultEdgeOptions={{ animated: false }}
    >
      <Background color="#334155" gap={20} />
      <Controls className="!bg-slate-800 !border-slate-700 !shadow-lg" />
      <MiniMap
        className="!bg-slate-800 !border-slate-700"
        nodeColor={(node) => {
          const data = node.data as { feature_color?: string; is_dead_code?: boolean };
          if (data?.is_dead_code) return '#ef4444';
          return data?.feature_color || '#475569';
        }}
      />
    </ReactFlow>
  );
}
