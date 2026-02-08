import { useState } from 'react';
import RepoForm from './RepoForm';
import RepoList from './RepoList';
import { useRepoStore } from '../stores/repoStore';

export default function Layout({ children }: { children: React.ReactNode }) {
  const [showForm, setShowForm] = useState(false);
  const { analysisStatus } = useRepoStore();

  return (
    <div className="h-screen flex flex-col bg-slate-950">
      {/* Header */}
      <header className="h-12 flex items-center justify-between px-4 bg-slate-900 border-b border-slate-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold text-slate-100">Code Flow Visualizer</h1>
          {analysisStatus && analysisStatus.status === 'running' && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span className="text-xs text-amber-400">
                {analysisStatus.current_stage} ({Math.round(analysisStatus.progress)}%)
              </span>
            </div>
          )}
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="px-3 py-1.5 text-xs rounded-lg bg-blue-600 text-white hover:bg-blue-500"
        >
          + Add Repo
        </button>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-slate-900 border-r border-slate-800 overflow-y-auto p-3 flex-shrink-0">
          {showForm && <RepoForm onSubmit={() => setShowForm(false)} />}
          <div className="mt-4">
            <RepoList />
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 relative">
          {children}
        </main>
      </div>
    </div>
  );
}
