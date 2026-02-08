import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { getNodeBorderColor, getNodeBgColor } from '../../utils/colors';

interface FunctionNodeData {
  label: string;
  file_path: string;
  node_type: string;
  description: string | null;
  is_entry_point: boolean;
  is_dead_code: boolean;
  feature_color: string | null;
  feature_name: string | null;
  language: string;
}

function FunctionNodeComponent({ data }: { data: FunctionNodeData }) {
  const borderColor = getNodeBorderColor(data.feature_color, data.is_dead_code, data.is_entry_point);
  const bgColor = getNodeBgColor(data.feature_color, data.is_dead_code);

  const typeIcon = {
    function: 'fn',
    method: 'M',
    class: 'C',
    file: 'F',
    module: 'Mod',
  }[data.node_type] || '?';

  return (
    <div
      className="px-3 py-2 rounded-lg shadow-lg min-w-[160px] max-w-[240px]"
      style={{
        backgroundColor: bgColor,
        border: `2px solid ${borderColor}`,
      }}
    >
      <Handle type="target" position={Position.Top} className="!bg-slate-500" />

      <div className="flex items-center gap-1.5 mb-1">
        <span
          className="text-[10px] font-bold px-1 py-0.5 rounded"
          style={{ backgroundColor: borderColor + '30', color: borderColor }}
        >
          {typeIcon}
        </span>
        {data.is_entry_point && (
          <span className="text-[10px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-400">
            EP
          </span>
        )}
        {data.is_dead_code && (
          <span className="text-[10px] px-1 py-0.5 rounded bg-red-500/20 text-red-400">
            DEAD
          </span>
        )}
        {data.feature_name && (
          <span
            className="text-[9px] px-1 py-0.5 rounded truncate max-w-[80px]"
            style={{ backgroundColor: data.feature_color + '20', color: data.feature_color || '#94a3b8' }}
          >
            {data.feature_name}
          </span>
        )}
      </div>

      <div className="text-sm font-mono font-medium text-slate-200 truncate">
        {data.label}
      </div>

      {data.description && (
        <div className="text-[10px] text-slate-400 mt-1 truncate">
          {data.description}
        </div>
      )}

      <div className="text-[9px] text-slate-500 mt-1 truncate">
        {data.file_path}
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-slate-500" />
    </div>
  );
}

export default memo(FunctionNodeComponent);
