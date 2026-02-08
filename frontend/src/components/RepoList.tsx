import { useRepoStore } from '../stores/repoStore';
import { useGraphStore } from '../stores/graphStore';

export default function RepoList() {
  const { repos, selectedRepoId, selectRepo, startAnalysis } = useRepoStore();
  const { fetchGraph, fetchDeadCode } = useGraphStore();

  const handleSelect = async (repoId: string) => {
    selectRepo(repoId);
    await fetchGraph(repoId);
    await fetchDeadCode(repoId);
  };

  const handleReanalyze = async (repoId: string) => {
    await startAnalysis(repoId);
  };

  if (repos.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
        Repositories
      </h3>
      {repos.map((repo) => (
        <div
          key={repo.id}
          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
            selectedRepoId === repo.id
              ? 'bg-blue-600/10 border-blue-500/30'
              : 'bg-slate-800/30 border-slate-700 hover:border-slate-600'
          }`}
          onClick={() => handleSelect(repo.id)}
        >
          <div className="text-sm font-medium text-slate-200 truncate">
            {repo.url || repo.local_path}
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-slate-500">{repo.branch}</span>
            {repo.last_analyzed_at && (
              <span className="text-xs text-green-400">analyzed</span>
            )}
          </div>
          {selectedRepoId === repo.id && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleReanalyze(repo.id);
              }}
              className="mt-2 text-xs px-3 py-1 rounded bg-slate-700 text-slate-300 hover:bg-slate-600"
            >
              Re-analyze
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
