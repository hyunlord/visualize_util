import type {
  RepoCreateRequest,
  RepoResponse,
  AnalysisStatusResponse,
  GraphResponse,
  DeadCodeResponse,
  FeatureFlowResponse,
} from './types';

const API_BASE = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  repos: {
    create: (data: RepoCreateRequest) =>
      request<RepoResponse>('/repos', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    list: () => request<RepoResponse[]>('/repos'),
    get: (id: string) => request<RepoResponse>(`/repos/${id}`),
    delete: (id: string) =>
      request<void>(`/repos/${id}`, { method: 'DELETE' }),
  },

  analysis: {
    start: (repoId: string, language: string = 'en') =>
      request<AnalysisStatusResponse>(`/repos/${repoId}/analyze`, {
        method: 'POST',
        body: JSON.stringify({ language }),
      }),
    status: (repoId: string) =>
      request<AnalysisStatusResponse>(`/repos/${repoId}/analyze`),
  },

  graph: {
    get: (repoId: string) =>
      request<GraphResponse>(`/repos/${repoId}/graph`),
  },

  deadCode: {
    get: (repoId: string) =>
      request<DeadCodeResponse>(`/repos/${repoId}/dead-code`),
  },

  features: {
    list: (repoId: string) =>
      request<GraphResponse['features']>(`/repos/${repoId}/features`),
    getFlow: (repoId: string, featureId: string) =>
      request<FeatureFlowResponse>(`/repos/${repoId}/features/${featureId}/flow`),
  },
};
