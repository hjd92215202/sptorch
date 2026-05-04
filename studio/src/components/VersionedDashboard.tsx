import { AlertTriangle, TrendingUp } from "lucide-react";
import React, { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
import { EvolutionMetrics, VersionNode } from "../types";

interface Props {
  metrics: EvolutionMetrics[];
  commits: VersionNode[];
}

function pctChange(prev: number, curr: number): number {
  if (Math.abs(prev) < 1e-8) return 0;
  return Math.abs((curr - prev) / prev);
}

export default function VersionedDashboard({ metrics, commits }: Props) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  const scaleAlert = useMemo(() => {
    if (metrics.length < 2) return false;
    const prev = metrics[metrics.length - 2].grad_scale_factor;
    const curr = metrics[metrics.length - 1].grad_scale_factor;
    return pctChange(prev, curr) > 0.3;
  }, [metrics]);

  const latest = metrics[metrics.length - 1];

  useEffect(() => {
    if (!chartRef.current) return;
    const chart = echarts.init(chartRef.current);

    const ts = metrics.map((m) => m.ts_ms);
    const loss = metrics.map((m) => Number(m.loss.toFixed(4)));
    const grad = metrics.map((m) => Number(m.grad_norm.toFixed(4)));
    const scale = metrics.map((m) => Number(m.grad_scale_factor.toFixed(4)));

    chart.setOption({
      animation: false,
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#dbeafe" } },
      grid: { left: 40, right: 16, top: 28, bottom: 28 },
      xAxis: {
        type: "category",
        data: ts,
        axisLabel: { color: "#94a3b8" },
        axisLine: { lineStyle: { color: "#334155" } }
      },
      yAxis: [
        {
          type: "value",
          name: "loss/grad",
          axisLabel: { color: "#94a3b8" },
          splitLine: { lineStyle: { color: "rgba(148,163,184,0.16)" } }
        },
        {
          type: "value",
          name: "scale",
          axisLabel: { color: "#94a3b8" },
          splitLine: { show: false }
        }
      ],
      series: [
        { name: "loss", type: "line", smooth: true, data: loss, lineStyle: { width: 2, color: "#67e8f9" } },
        { name: "grad_norm", type: "line", smooth: true, data: grad, lineStyle: { width: 2, color: "#a78bfa" } },
        {
          name: "grad_scale_factor",
          type: "line",
          smooth: true,
          yAxisIndex: 1,
          data: scale,
          lineStyle: { width: 2, color: "#f59e0b" }
        }
      ]
    });

    return () => chart.dispose();
  }, [metrics]);

  const accumPct = latest ? Math.min(100, (latest.accum_current / Math.max(1, latest.accum_target)) * 100) : 0;

  return (
    <section className="rounded-2xl border border-slate-700/50 bg-slate-900/50 p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp size={18} className="text-cyan-300" />
          <h2 className="text-sm font-semibold text-slate-100">Versioned Dashboard</h2>
        </div>
        {scaleAlert && (
          <span className="inline-flex items-center gap-1 rounded-full border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-xs text-amber-300">
            <AlertTriangle size={14} />
            scale_gradients 波动&gt;30%
          </span>
        )}
      </div>

      <div ref={chartRef} className="h-64 w-full" />

      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-slate-700/40 bg-slate-950/50 p-3">
          <div className="mb-2 text-xs text-slate-400">梯度累积进度</div>
          <div className="mb-2 text-sm text-slate-200">
            {latest ? `${latest.accum_current} / ${latest.accum_target}` : "0 / 0"}
          </div>
          <div className="h-2 w-full rounded-full bg-slate-800">
            <div className="h-2 rounded-full bg-cyan-400 transition-all" style={{ width: `${accumPct}%` }} />
          </div>
        </div>

        <div className="rounded-xl border border-slate-700/40 bg-slate-950/50 p-3">
          <div className="mb-2 text-xs text-slate-400">Version Timeline</div>
          <div className="max-h-24 space-y-1 overflow-auto text-xs text-slate-200">
            {commits.slice(-8).map((c) => (
              <div key={`${c.version_id}-${c.committed_at_ms}`} className="flex justify-between gap-2">
                <span>v{c.version_id} ({c.reason})</span>
                <span className="text-slate-400">{c.committed_at_ms}</span>
              </div>
            ))}
            {commits.length === 0 && <div className="text-slate-500">No commit yet</div>}
          </div>
        </div>
      </div>
    </section>
  );
}
