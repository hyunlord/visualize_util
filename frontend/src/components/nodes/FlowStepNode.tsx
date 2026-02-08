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
      className="rounded-xl shadow-lg border-2 bg-slate-800/90 backdrop-blur-sm overflow-hidden"
      style={{
        borderColor,
        minWidth: '280px',
        maxWidth: '340px',
      }}
    >
      <Handle type="target" position={Position.Top} className="!bg-slate-500 !w-2.5 !h-2.5" />

      {/* Header */}
      <div
        className="px-4 py-2 flex items-center gap-2"
        style={{ backgroundColor: borderColor + '15' }}
      >
        {/* Step number */}
        <span
          className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
          style={{ backgroundColor: borderColor }}
        >
          {data.order}
        </span>

        {isEntry && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-medium">
            ENTRY
          </span>
        )}

        {/* Function name */}
        <span className="text-sm font-mono font-semibold text-slate-100 truncate">
          {data.label}
        </span>
      </div>

      {/* Body */}
      <div className="px-4 py-3">
        {/* Description */}
        <p className="text-xs text-slate-300 leading-relaxed">
          {data.description}
        </p>

        {/* File path */}
        <div className="text-[10px] text-slate-500 mt-2 font-mono truncate">
          {data.file_path}
          {data.line_start && (
            <span className="text-slate-600">
              :{data.line_start}
              {data.line_end && data.line_end !== data.line_start ? `-${data.line_end}` : ''}
            </span>
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-slate-500 !w-2.5 !h-2.5" />
    </div>
  );
}

export default memo(FlowStepNodeComponent);
