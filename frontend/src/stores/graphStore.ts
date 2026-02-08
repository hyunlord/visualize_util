import { create } from 'zustand';
import type { GraphResponse, GraphNode, GraphEdge, FeatureInfo, DeadCodeItem } from '../api/types';
import { api } from '../api/client';

interface GraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  features: FeatureInfo[];
  deadCode: DeadCodeItem[];
  selectedNodeId: string | null;
  activeFeatureIds: Set<string>;
  loading: boolean;
  error: string | null;

  fetchGraph: (repoId: string) => Promise<void>;
  fetchDeadCode: (repoId: string) => Promise<void>;
  selectNode: (nodeId: string | null) => void;
  toggleFeature: (featureId: string) => void;
  showAllFeatures: () => void;
  showOnlyFeature: (featureId: string) => void;
  getFilteredNodes: () => GraphNode[];
  getFilteredEdges: () => GraphEdge[];
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  features: [],
  deadCode: [],
  selectedNodeId: null,
  activeFeatureIds: new Set(),
  loading: false,
  error: null,

  fetchGraph: async (repoId: string) => {
    try {
      set({ loading: true, error: null });
      const data: GraphResponse = await api.graph.get(repoId);
      const allFeatureIds = new Set(data.features.map((f) => f.id));
      set({
        nodes: data.nodes,
        edges: data.edges,
        features: data.features,
        activeFeatureIds: allFeatureIds,
        loading: false,
      });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  fetchDeadCode: async (repoId: string) => {
    try {
      const data = await api.deadCode.get(repoId);
      set({ deadCode: data.items });
    } catch {
      // Non-critical
    }
  },

  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
  },

  toggleFeature: (featureId: string) => {
    set((state) => {
      const newSet = new Set(state.activeFeatureIds);
      if (newSet.has(featureId)) {
        newSet.delete(featureId);
      } else {
        newSet.add(featureId);
      }
      return { activeFeatureIds: newSet };
    });
  },

  showAllFeatures: () => {
    set((state) => ({
      activeFeatureIds: new Set(state.features.map((f) => f.id)),
    }));
  },

  showOnlyFeature: (featureId: string) => {
    set({ activeFeatureIds: new Set([featureId]) });
  },

  getFilteredNodes: () => {
    const { nodes, activeFeatureIds } = get();
    return nodes.filter(
      (n) =>
        !n.data.feature_id || activeFeatureIds.has(n.data.feature_id)
    );
  },

  getFilteredEdges: () => {
    const { edges } = get();
    const filteredNodeIds = new Set(get().getFilteredNodes().map((n) => n.id));
    return edges.filter(
      (e) => filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target)
    );
  },
}));
