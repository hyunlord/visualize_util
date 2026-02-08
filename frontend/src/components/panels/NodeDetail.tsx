import { useGraphStore } from '../../stores/graphStore';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import { useEffect, useRef } from 'react';

export default function NodeDetail() {
  const { nodes, selectedNodeId, selectNode } = useGraphStore();
  const codeRef = useRef<HTMLElement>(null);

  const node = nodes.find((n) => n.id === selectedNodeId);

  useEffect(() => {
    if (codeRef.current) {
      Prism.highlightElement(codeRef.current);
    }
  }, [node?.data.source_code]);

  if (!node) return null;

  const { data } = node;
  const langClass = {
    python: 'language-python',
    javascript: 'language-javascript',
    typescript: 'language-typescript',
  }[data.language] || 'language-python';

  return (
    <div className="absolute right-0 top-0 h-full w-96 bg-slate-900 border-l border-slate-700 overflow-y-auto z-10 shadow-2xl">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-100">{data.label}</h3>
          <button
            onClick={() => selectNode(null)}
            className="text-slate-400 hover:text-slate-200 text-xl"
          >
            &times;
          </button>
        </div>

        <div className="space-y-3 text-sm">
          <div className="flex gap-2 flex-wrap">
            <span className="px-2 py-1 rounded bg-slate-800 text-slate-300 text-xs">
              {data.node_type}
            </span>
            <span className="px-2 py-1 rounded bg-slate-800 text-slate-300 text-xs">
              {data.language}
            </span>
            {data.is_entry_point && (
              <span className="px-2 py-1 rounded bg-amber-500/20 text-amber-400 text-xs">
                Entry Point
              </span>
            )}
            {data.is_dead_code && (
              <span className="px-2 py-1 rounded bg-red-500/20 text-red-400 text-xs">
                Dead Code
              </span>
            )}
          </div>

          <div>
            <span className="text-slate-500">File: </span>
            <span className="text-slate-300 font-mono text-xs">{data.file_path}</span>
          </div>

          <div>
            <span className="text-slate-500">Lines: </span>
            <span className="text-slate-300">{data.line_start} - {data.line_end}</span>
          </div>

          {data.feature_name && (
            <div className="flex items-center gap-2">
              <span className="text-slate-500">Feature: </span>
              <span
                className="px-2 py-0.5 rounded text-xs"
                style={{
                  backgroundColor: (data.feature_color || '#475569') + '20',
                  color: data.feature_color || '#94a3b8',
                }}
              >
                {data.feature_name}
              </span>
            </div>
          )}

          {data.description && (
            <div className="p-3 rounded bg-slate-800/50 border border-slate-700">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
                AI Description
              </div>
              <div className="text-slate-300 text-sm">{data.description}</div>
            </div>
          )}

          {data.docstring && (
            <div className="p-3 rounded bg-slate-800/50 border border-slate-700">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
                Docstring
              </div>
              <div className="text-slate-400 text-xs font-mono whitespace-pre-wrap">
                {data.docstring}
              </div>
            </div>
          )}

          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">
              Source Code
            </div>
            <pre className="rounded bg-slate-950 p-3 overflow-x-auto text-xs max-h-96">
              <code ref={codeRef} className={langClass}>
                {data.source_code}
              </code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
