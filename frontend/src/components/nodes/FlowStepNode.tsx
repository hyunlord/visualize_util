import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

interface FlowStepNodeData {
  label: string;
  file_path: string;
  description: string;
  order: number;
  is_entry_point: boolean;
  source_code: string | null;
  feature_color: string;
  line_start: number | null;
  line_end: number | null;
  [key: string]: unknown;
}

function FlowStepNodeComponent({ data }: { data: FlowStepNodeData }) {
  const isEntry = data.is_entry_point || data.order === 1;
  const borderColor = isEntry ? '#f59e0b' : data.feature_color || '#3b82f6';

  return (
    <div
      className="rounded-xl shadow-lg border-2 bg-slate-800/90 backdrop-blur-sm overflow-hidden cursor-pointer hover:brightness-110 transition-all"
      style={{
        borderColor,
        minWidth: '360px',
        maxWidth: '440px',
      }}
    >
      <Handle type="target" position={Position.Top} className="!bg-slate-500 !w-3 !h-3" />

      {/* Header */}
      <div
        className="px-4 py-2.5 flex items-center gap-2.5"
        style={{ backgroundColor: borderColor + '15' }}
      >
        {/* Step number */}
        <span
          className="w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold text-white flex-shrink-0"
          style={{ backgroundColor: borderColor }}
        >
          {data.order}
        </span>

        {isEntry && (
          <span className="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 font-medium">
            ENTRY
          </span>
        )}

        {/* Function name */}
        <span className="text-base font-mono font-semibold text-slate-100 truncate">
          {data.label}
        </span>
      </div>

      {/* Body */}
      <div className="px-4 py-3.5">
        {/* Description */}
        <p className="text-sm text-slate-300 leading-relaxed">
          {data.description}
        </p>

        {/* File path */}
        <div className="text-xs text-slate-500 mt-2.5 font-mono truncate">
          {data.file_path}
          {data.line_start && (
            <span className="text-slate-600">
              :{data.line_start}
              {data.line_end && data.line_end !== data.line_start ? `-${data.line_end}` : ''}
            </span>
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-slate-500 !w-3 !h-3" />
    </div>
  );
}

export default memo(FlowStepNodeComponent);
