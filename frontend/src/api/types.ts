export interface RepoCreateRequest {
  url?: string;
  local_path?: string;
  branch?: string;
}

export interface RepoResponse {
  id: string;
  url: string | null;
  local_path: string;
  branch: string;
  last_commit_sha: string | null;
  last_analyzed_at: string | null;
  created_at: string;
}

export interface AnalysisStatusResponse {
  snapshot_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  current_stage: string;
}

export interface GraphNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    label: string;
    file_path: string;
    node_type: string;
    language: string;
    line_start: number;
    line_end: number;
    source_code: string;
    docstring: string | null;
    description: string | null;
    is_entry_point: boolean;
    is_dead_code: boolean;
    feature_id: string | null;
    feature_color: string | null;
    feature_name: string | null;
  };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  animated: boolean;
  data: {
    edge_type: string;
    line_number: number | null;
    is_llm_inferred: boolean;
  };
}

export interface FeatureInfo {
  id: string;
  name: string;
  description: string | null;
  color: string;
  node_count: number;
  auto_detected: boolean;
  flow_summary?: string | null;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  features: FeatureInfo[];
}

export interface FlowStep {
  order: number;
  node_id: string | null;
  file: string;
  function: string;
  description: string;
  source_code: string | null;
  line_start: number | null;
  line_end: number | null;
  calls_next: string[];
}

export interface FeatureFlowResponse {
  feature: FeatureInfo;
  flow_summary: string | null;
  flow_steps: FlowStep[];
  edges: GraphEdge[];
}

export interface DeadCodeItem {
  node_id: string;
  file_path: string;
  name: string;
  node_type: string;
  line_start: number;
  line_end: number;
  reason: string;
  confidence: number;
  llm_explanation: string | null;
  suggested_feature: string | null;
}

export interface DeadCodeResponse {
  items: DeadCodeItem[];
  total_count: number;
}
