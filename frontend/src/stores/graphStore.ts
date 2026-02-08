import { create } from 'zustand';
import type {
  GraphResponse,
  GraphNode,
  GraphEdge,
  FeatureInfo,
  DeadCodeItem,
  FeatureFlowResponse,
} from '../api/types';
import { api } from '../api/client';

type ViewMode = 'overview' | 'flow';

interface GraphState {
  // View navigation
  currentView: ViewMode;
  selectedFeatureId: string | null;

  // Data
  nodes: GraphNode[];
  edges: GraphEdge[];
  features: FeatureInfo[];
  deadCode: DeadCodeItem[];
  featureFlow: FeatureFlowResponse | null;

  // UI state
  selectedNodeId: string | null;
  activeFeatureIds: Set<string>;
  loading: boolean;
  flowLoading: boolean;
  error: string | null;

  // Actions
  fetchGraph: (repoId: string) => Promise<void>;
  fetchDeadCode: (repoId: string) => Promise<void>;
  fetchFeatureFlow: (repoId: string, featureId: string) => Promise<void>;
  selectNode: (nodeId: string | null) => void;
  selectFeature: (featureId: string) => void;
  goToOverview: () => void;
  toggleFeature: (featureId: string) => void;
  showAllFeatures: () => void;
  showOnlyFeature: (featureId: string) => void;
  getFilteredNodes: () => GraphNode[];
  getFilteredEdges: () => GraphEdge[];
}

export const useGraphStore = create<GraphState>((set, get) => ({
  currentView: 'overview',
  selectedFeatureId: null,

  nodes: [],
  edges: [],
  features: [],
  deadCode: [],
  featureFlow: null,

  selectedNodeId: null,
  activeFeatureIds: new Set(),
  loading: false,
  flowLoading: false,
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
        currentView: 'overview',
        selectedFeatureId: null,
        featureFlow: null,
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

  fetchFeatureFlow: async (repoId: string, featureId: string) => {
    try {
      set({ flowLoading: true, error: null });
      const data = await api.features.getFlow(repoId, featureId);
      set({
        featureFlow: data,
        flowLoading: false,
        currentView: 'flow',
        selectedFeatureId: featureId,
      });
    } catch (e) {
      set({ error: (e as Error).message, flowLoading: false });
    }
  },

  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
  },

  selectFeature: (featureId: string) => {
    set({ selectedFeatureId: featureId, currentView: 'flow' });
  },

  goToOverview: () => {
    set({
      currentView: 'overview',
      selectedFeatureId: null,
      featureFlow: null,
      selectedNodeId: null,
    });
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
      (n) => !n.data.feature_id || activeFeatureIds.has(n.data.feature_id)
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
