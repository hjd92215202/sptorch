import { MemoryStick } from "lucide-react";
import React, { useMemo, useState } from "react";
import { TensorLayoutSnapshot } from "../types";

interface Props {
  tensors: TensorLayoutSnapshot[];
}

function strideBars(strides: number[]) {
  const max = Math.max(...strides, 1);
  return strides.map((s, idx) => ({ idx, ratio: Math.max(4, (s / max) * 100), val: s }));
}

export default function MemorySnapshotPanel({ tensors }: Props) {
  const [selected, setSelected] = useState<string | null>(tensors[0]?.tensor_id ?? null);
  React.useEffect(() => {
    if (!selected && tensors.length > 0) {
      setSelected(tensors[0].tensor_id);
      return;
    }
    if (selected && !tensors.some((t) => t.tensor_id === selected)) {
      setSelected(tensors[0]?.tensor_id ?? null);
    }
  }, [selected, tensors]);

  const selectedTensor = useMemo(() => tensors.find((t) => t.tensor_id === selected), [selected, tensors]);

  return (
    <section className="rounded-2xl border border-slate-700/50 bg-slate-900/50 p-4">
      <div className="mb-3 flex items-center gap-2">
        <MemoryStick size={18} className="text-violet-300" />
        <h2 className="text-sm font-semibold text-slate-100">Tensor Memory Map</h2>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[280px_1fr]">
        <div className="max-h-72 space-y-1 overflow-auto rounded-xl border border-slate-700/40 bg-slate-950/50 p-2">
          {tensors.map((t) => (
            <button
              key={t.tensor_id}
              onClick={() => setSelected(t.tensor_id)}
              className={`w-full rounded-lg px-2 py-2 text-left text-xs transition ${
                selected === t.tensor_id
                  ? "bg-cyan-500/20 text-cyan-100"
                  : "bg-slate-900/40 text-slate-300 hover:bg-slate-800/70"
              }`}
            >
              <div className="font-medium">{t.tensor_id}</div>
              <div className="text-[11px] text-slate-400">shape [{t.shape.join(", ")}] · {t.device}</div>
            </button>
          ))}
        </div>

        <div className="rounded-xl border border-slate-700/40 bg-slate-950/50 p-3 text-xs text-slate-200">
          {!selectedTensor && <div className="text-slate-500">No tensor selected</div>}
          {selectedTensor && (
            <>
              <div className="mb-2 font-medium text-cyan-200">{selectedTensor.tensor_id}</div>
              <div className="mb-3 grid grid-cols-2 gap-2 text-[11px] text-slate-300">
                <div>shape: [{selectedTensor.shape.join(", ")}]</div>
                <div>strides: [{selectedTensor.strides.join(", ")}]</div>
                <div>offset: {selectedTensor.offset}</div>
                <div>numel: {selectedTensor.numel}</div>
                <div>dtype: {selectedTensor.dtype}</div>
                <div>device: {selectedTensor.device}</div>
              </div>

              <div className="mb-2 text-[11px] text-slate-400">Physical Layout by Stride</div>
              <div className="space-y-2">
                {strideBars(selectedTensor.strides).map((b) => (
                  <div key={b.idx} className="flex items-center gap-2">
                    <span className="w-16 text-[11px] text-slate-400">dim {b.idx}</span>
                    <div className="h-2 flex-1 rounded-full bg-slate-800">
                      <div className="h-2 rounded-full bg-violet-400" style={{ width: `${b.ratio}%` }} />
                    </div>
                    <span className="w-10 text-right text-[11px] text-slate-300">{b.val}</span>
                  </div>
                ))}
              </div>

              <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2">
                <div className="rounded border border-cyan-500/30 bg-cyan-500/10 p-2">
                  <div className="text-[11px] text-cyan-200">active_ptr</div>
                  <div className="break-all font-mono text-[11px] text-cyan-100">{selectedTensor.pointers.active_ptr}</div>
                </div>
                <div className="rounded border border-fuchsia-500/30 bg-fuchsia-500/10 p-2">
                  <div className="text-[11px] text-fuchsia-200">shadow_ptr</div>
                  <div className="break-all font-mono text-[11px] text-fuchsia-100">
                    {selectedTensor.pointers.shadow_ptr ?? "<single-buffer>"}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
