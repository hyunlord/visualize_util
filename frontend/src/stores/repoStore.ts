import { create } from 'zustand';
import type { RepoResponse, AnalysisStatusResponse } from '../api/types';
import { api } from '../api/client';

interface RepoState {
  repos: RepoResponse[];
  selectedRepoId: string | null;
  analysisStatus: AnalysisStatusResponse | null;
  analysisLanguage: string;
  loading: boolean;
  error: string | null;

  fetchRepos: () => Promise<void>;
  addRepo: (url?: string, localPath?: string, branch?: string) => Promise<RepoResponse>;
  selectRepo: (id: string) => void;
  setAnalysisLanguage: (lang: string) => void;
  startAnalysis: (repoId: string) => Promise<void>;
  pollAnalysisStatus: (repoId: string) => Promise<void>;
  clearError: () => void;
}

export const useRepoStore = create<RepoState>((set, get) => ({
  repos: [],
  selectedRepoId: null,
  analysisStatus: null,
  analysisLanguage: 'en',
  loading: false,
  error: null,

  fetchRepos: async () => {
    try {
      set({ loading: true, error: null });
      const repos = await api.repos.list();
      set({ repos, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  addRepo: async (url?: string, localPath?: string, branch?: string) => {
    set({ loading: true, error: null });
    try {
      const repo = await api.repos.create({ url, local_path: localPath, branch });
      set((state) => ({ repos: [...state.repos, repo], loading: false }));
      return repo;
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
      throw e;
    }
  },

  selectRepo: (id: string) => {
    set({ selectedRepoId: id, analysisStatus: null });
  },

  setAnalysisLanguage: (lang: string) => {
    set({ analysisLanguage: lang });
  },

  startAnalysis: async (repoId: string) => {
    try {
      set({ loading: true, error: null });
      const language = get().analysisLanguage;
      const status = await api.analysis.start(repoId, language);
      set({ analysisStatus: status, loading: false });
      if (status.status === 'running') {
        get().pollAnalysisStatus(repoId);
      }
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  pollAnalysisStatus: async (repoId: string) => {
    const poll = async () => {
      try {
        const status = await api.analysis.status(repoId);
        set({ analysisStatus: status });
        if (status.status === 'running') {
          setTimeout(poll, 2000);
        }
      } catch {
        // Polling failed, stop
      }
    };
    setTimeout(poll, 2000);
  },

  clearError: () => set({ error: null }),
}));
