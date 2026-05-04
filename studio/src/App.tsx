import { Activity, Layers3 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  getEngineStatus,
  onFence,
  onHardware,
  onMetrics,
  onVersionCommit,
  startEvolutionStream,
  triggerAtomicSwap
} from "./api";
import AutogradVersionGraph from "./components/AutogradVersionGraph";
import HardwareFencePanel from "./components/HardwareFencePanel";
import MemorySnapshotPanel from "./components/MemorySnapshotPanel";
import VersionedDashboard from "./components/VersionedDashboard";
import { EngineStatusDto, EvolutionMetrics, FenceState, HardwareState, VersionNode } from "./types";

export default function App() {
  const [status, setStatus] = useState<EngineStatusDto | null>(null);
  const [metrics, setMetrics] = useState<EvolutionMetrics[]>([]);
  const [commits, setCommits] = useState<VersionNode[]>([]);
  const [fence, setFence] = useState<FenceState | null>(null);
  const [hardware, setHardware] = useState<HardwareState | null>(null);
  const [swapping, setSwapping] = useState(false);

  useEffect(() => {
    let unlistenMetrics: (() => void) | undefined;
    let unlistenCommit: (() => void) | undefined;
    let unlistenFence: (() => void) | undefined;
    let unlistenHardware: (() => void) | undefined;

    (async () => {
      const s = await getEngineStatus();
      setStatus(s);
      if (s.chain_head) setCommits([s.chain_head]);

      await startEvolutionStream();

      unlistenMetrics = await onMetrics((m) => {
        setMetrics((prev) => [...prev.slice(-239), m]);
      });

      unlistenCommit = await onVersionCommit((v) => {
        setCommits((prev) => [...prev.slice(-31), v]);
        setStatus((prev) =>
          prev
            ? {
                ...prev,
                global_version: Math.max(prev.global_version, v.version_id),
                active_version: v.version_id,
                chain_head: v
              }
            : prev
        );
      });

      unlistenFence = await onFence((f) => {
        setFence(f);
        if (f.phase === "Done" || f.phase === "Error") {
          setSwapping(false);
        }
      });

      unlistenHardware = await onHardware((h) => {
        setHardware(h);
      });
    })();

    return () => {
      unlistenMetrics?.();
      unlistenCommit?.();
      unlistenFence?.();
      unlistenHardware?.();
    };
  }, []);

  const policyStats = useMemo(() => {
    const policies = status?.layer_policies ?? [];
    const single = policies.filter((p) => p.policy === "Single").length;
    const double = policies.filter((p) => p.policy === "Double").length;
    return { single, double, total: policies.length };
  }, [status]);

  return (
    <main className="mx-auto flex h-full max-w-[1600px] flex-col gap-4 p-4">
      <header className="rounded-2xl border border-slate-700/50 bg-[var(--panel)] p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-slate-100">SPTorch Studio v1</h1>
            <p className="text-sm text-slate-400">Versioned Tensor Protocol Control Center</p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full border border-cyan-400/40 bg-cyan-500/10 px-2 py-1 text-cyan-200">
              global v{status?.global_version ?? "-"}
            </span>
            <span className="rounded-full border border-violet-400/40 bg-violet-500/10 px-2 py-1 text-violet-200">
              active v{status?.active_version ?? "-"}
            </span>
            <span className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-2 py-1 text-emerald-200">
              layers {policyStats.total}
            </span>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-1 gap-2 text-xs md:grid-cols-3">
          <div className="rounded-lg border border-slate-700/40 bg-slate-900/50 p-2 text-slate-300">
            <div className="mb-1 flex items-center gap-1 text-slate-400">
              <Layers3 size={13} /> UpdatePolicy
            </div>
            <div>Single Buffer: {policyStats.single}</div>
            <div>Double Buffer: {policyStats.double}</div>
          </div>
          <div className="rounded-lg border border-slate-700/40 bg-slate-900/50 p-2 text-slate-300">
            <div className="mb-1 flex items-center gap-1 text-slate-400">
              <Activity size={13} /> Metrics Stream
            </div>
            <div>samples: {metrics.length}</div>
            <div>latest loss: {metrics.at(-1)?.loss?.toFixed(4) ?? "-"}</div>
          </div>
          <div className="rounded-lg border border-slate-700/40 bg-slate-900/50 p-2 text-slate-300">
            <div className="mb-1 text-slate-400">Hardware</div>
            <div>backend: {hardware?.backend ?? "simulated-hal-ffi"}</div>
            <div>online: {String(hardware?.online ?? true)}</div>
          </div>
        </div>
      </header>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <VersionedDashboard metrics={metrics} commits={commits} />
        <HardwareFencePanel
          fence={fence}
          hardware={hardware}
          swapping={swapping}
          onSwap={async () => {
            setSwapping(true);
            await triggerAtomicSwap();
          }}
        />
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <MemorySnapshotPanel tensors={status?.tensors ?? []} />
        <AutogradVersionGraph commits={commits} />
      </section>
    </main>
  );
}
