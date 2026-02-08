import { useState } from 'react';
import { useRepoStore } from '../stores/repoStore';

export default function RepoForm({ onSubmit }: { onSubmit: () => void }) {
  const [inputType, setInputType] = useState<'url' | 'local'>('local');
  const [url, setUrl] = useState('');
  const [localPath, setLocalPath] = useState('');
  const [branch, setBranch] = useState('main');
  const { addRepo, startAnalysis, loading, error, clearError } = useRepoStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();

    try {
      const repo = await addRepo(
        inputType === 'url' ? url : undefined,
        inputType === 'local' ? localPath : undefined,
        branch
      );
      await startAnalysis(repo.id);
      onSubmit();
    } catch {
      // Error handled in store
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      <h2 className="text-xl font-semibold text-slate-100">Analyze Repository</h2>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setInputType('local')}
          className={`px-4 py-2 rounded-lg text-sm ${
            inputType === 'local'
              ? 'bg-blue-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          Local Path
        </button>
        <button
          type="button"
          onClick={() => setInputType('url')}
          className={`px-4 py-2 rounded-lg text-sm ${
            inputType === 'url'
              ? 'bg-blue-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          GitHub URL
        </button>
      </div>

      {inputType === 'url' ? (
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://github.com/user/repo"
          className="w-full px-4 py-2 rounded-lg bg-slate-900 border border-slate-600 text-slate-200 placeholder-slate-500 focus:border-blue-500 focus:outline-none"
        />
      ) : (
        <input
          type="text"
          value={localPath}
          onChange={(e) => setLocalPath(e.target.value)}
          placeholder="/path/to/your/project"
          className="w-full px-4 py-2 rounded-lg bg-slate-900 border border-slate-600 text-slate-200 placeholder-slate-500 focus:border-blue-500 focus:outline-none"
        />
      )}

      <input
        type="text"
        value={branch}
        onChange={(e) => setBranch(e.target.value)}
        placeholder="Branch (default: main)"
        className="w-full px-4 py-2 rounded-lg bg-slate-900 border border-slate-600 text-slate-200 placeholder-slate-500 focus:border-blue-500 focus:outline-none"
      />

      {error && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Analyzing...' : 'Start Analysis'}
      </button>
    </form>
  );
}
