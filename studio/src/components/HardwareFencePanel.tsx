import { Cpu, Gauge } from "lucide-react";
import React from "react";
import { FenceState, HardwareState } from "../types";

interface Props {
  fence?: FenceState | null;
  hardware?: HardwareState | null;
  onSwap: () => void;
  swapping: boolean;
}

const ordered = ["Prepare", "WaitFence", "Swap", "Commit", "Done"];

export default function HardwareFencePanel({ fence, hardware, onSwap, swapping }: Props) {
  const phase = fence?.phase ?? "Idle";

  return (
    <section className="rounded-2xl border border-slate-700/50 bg-slate-900/50 p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu size={18} className="text-emerald-300" />
          <h2 className="text-sm font-semibold text-slate-100">Hardware Fence Panel</h2>
        </div>
        <button
          onClick={onSwap}
          disabled={swapping}
          className="rounded-lg border border-emerald-400/40 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-200 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {swapping ? "Switching..." : "物理切换（模拟）"}
        </button>
      </div>

      <div className="mb-3 rounded-xl border border-slate-700/40 bg-slate-950/50 p-3 text-xs text-slate-200">
        <div className="mb-2 flex items-center gap-2">
          <Gauge size={14} className="text-amber-300" />
          <span>Queue Depth: {hardware?.queue_depth ?? fence?.queue_depth ?? 0}</span>
          <span className="ml-auto text-slate-400">backend: {hardware?.backend ?? "simulated-hal-ffi"}</span>
        </div>
        <div className="h-2 w-full rounded-full bg-slate-800">
          <div className="h-2 rounded-full bg-emerald-400" style={{ width: `${Math.round((fence?.progress ?? 0) * 100)}%` }} />
        </div>
        <div className="mt-1 text-[11px] text-slate-400">
          phase: {phase} · {fence?.message ?? "idle"}
        </div>
      </div>

      <div className="grid grid-cols-5 gap-2 text-center text-[11px]">
        {ordered.map((p) => {
          const active = p === phase;
          const done = ordered.indexOf(p) < ordered.indexOf(String(phase));
          return (
            <div
              key={p}
              className={`rounded-lg border px-1 py-2 ${
                active
                  ? "border-cyan-400/60 bg-cyan-500/20 text-cyan-100"
                  : done
                  ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-200"
                  : "border-slate-700/50 bg-slate-900/60 text-slate-400"
              }`}
            >
              {p}
            </div>
          );
        })}
      </div>
    </section>
  );
}
