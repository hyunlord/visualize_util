import { useEffect, useState } from 'react';
import Layout from './components/Layout';
import FlowGraph from './components/FlowGraph';
import NodeDetail from './components/panels/NodeDetail';
import DeadCodePanel from './components/panels/DeadCodePanel';
import FeaturePanel from './components/panels/FeaturePanel';
import { useRepoStore } from './stores/repoStore';
import { useGraphStore } from './stores/graphStore';

export default function App() {
  const { fetchRepos, selectedRepoId } = useRepoStore();
  const { nodes, selectedNodeId, deadCode, features } = useGraphStore();
  const [showDeadCode, setShowDeadCode] = useState(false);
  const [showFeatures, setShowFeatures] = useState(false);

  useEffect(() => {
    fetchRepos();
  }, [fetchRepos]);

  return (
    <Layout>
      {nodes.length > 0 ? (
        <>
          {/* Toolbar */}
          <div className="absolute top-2 left-2 z-20 flex gap-2">
            <button
              onClick={() => { setShowFeatures(!showFeatures); setShowDeadCode(false); }}
              className={`px-3 py-1.5 text-xs rounded-lg border ${
                showFeatures
                  ? 'bg-blue-600/20 border-blue-500/40 text-blue-400'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Features ({features.length})
            </button>
            <button
              onClick={() => { setShowDeadCode(!showDeadCode); setShowFeatures(false); }}
              className={`px-3 py-1.5 text-xs rounded-lg border ${
                showDeadCode
                  ? 'bg-red-600/20 border-red-500/40 text-red-400'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
            >
              Dead Code ({deadCode.length})
            </button>
          </div>

          {/* Panels */}
          {showFeatures && <FeaturePanel onClose={() => setShowFeatures(false)} />}
          {showDeadCode && <DeadCodePanel onClose={() => setShowDeadCode(false)} />}
          {selectedNodeId && <NodeDetail />}

          {/* Graph */}
          <FlowGraph />
        </>
      ) : (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="text-6xl mb-4 opacity-20">&#x1f4ca;</div>
            <h2 className="text-xl font-semibold text-slate-400 mb-2">
              {selectedRepoId
                ? 'Analyzing repository...'
                : 'No repository selected'}
            </h2>
            <p className="text-sm text-slate-500">
              {selectedRepoId
                ? 'Please wait while the code is being analyzed.'
                : 'Add a repository using the + button above to get started.'}
            </p>
          </div>
        </div>
      )}
    </Layout>
  );
}
