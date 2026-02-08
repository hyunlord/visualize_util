import { useEffect } from 'react';
import Layout from './components/Layout';
import FeatureOverview from './views/FeatureOverview';
import FeatureFlowView from './views/FeatureFlowView';
import FlowNodeDetail from './components/panels/FlowNodeDetail';
import { useRepoStore } from './stores/repoStore';
import { useGraphStore } from './stores/graphStore';

export default function App() {
  const { fetchRepos, selectedRepoId } = useRepoStore();
  const {
    features,
    currentView,
    featureFlow,
    flowLoading,
    selectedNodeId,
    loading,
    fetchFeatureFlow,
    goToOverview,
    selectNode,
  } = useGraphStore();

  useEffect(() => {
    fetchRepos();
  }, [fetchRepos]);

  const handleSelectFeature = (featureId: string) => {
    if (selectedRepoId) {
      fetchFeatureFlow(selectedRepoId, featureId);
    }
  };

  const renderContent = () => {
    // Loading state
    if (loading) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm text-slate-400">Loading graph data...</p>
          </div>
        </div>
      );
    }

    // No features yet
    if (features.length === 0) {
      return (
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
      );
    }

    // Feature Flow View
    if (currentView === 'flow' && featureFlow) {
      return (
        <>
          <FeatureFlowView
            flowData={featureFlow}
            onBack={goToOverview}
            onNodeClick={selectNode}
          />
          {selectedNodeId && <FlowNodeDetail />}
        </>
      );
    }

    // Flow loading state
    if (flowLoading) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm text-slate-400">Loading feature flow...</p>
          </div>
        </div>
      );
    }

    // Feature Overview (default)
    return (
      <FeatureOverview
        features={features}
        onSelectFeature={handleSelectFeature}
      />
    );
  };

  return (
    <Layout>
      {renderContent()}
    </Layout>
  );
}
