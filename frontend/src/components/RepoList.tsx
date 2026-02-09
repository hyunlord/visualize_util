import { useRepoStore } from '../stores/repoStore';
import { useGraphStore } from '../stores/graphStore';

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'ko', label: '한국어' },
];

export default function RepoList() {
  const { repos, selectedRepoId, selectRepo, startAnalysis, analysisLanguage, setAnalysisLanguage } = useRepoStore();
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
            <div className="mt-2 flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
              <select
                value={analysisLanguage}
                onChange={(e) => setAnalysisLanguage(e.target.value)}
                className="text-xs px-2 py-1 rounded bg-slate-700 text-slate-300 border border-slate-600 focus:outline-none focus:border-blue-500"
              >
                {LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.label}
                  </option>
                ))}
              </select>
              <button
                onClick={() => handleReanalyze(repo.id)}
                className="text-xs px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-500 transition-colors"
              >
                Analyze
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
