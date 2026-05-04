import { GitBranch } from "lucide-react";
import React, { useMemo, useState } from "react";
import ReactFlow, { Controls, MiniMap, Node, Edge } from "reactflow";
import "reactflow/dist/style.css";
import { VersionNode } from "../types";

interface Props {
  commits: VersionNode[];
}

export default function AutogradVersionGraph({ commits }: Props) {
  const [selected, setSelected] = useState<VersionNode | null>(null);

  const nodes = useMemo<Node[]>(() => {
    const data = commits.length > 0 ? commits.slice(-6) : [{ version_id: 1, reason: "bootstrap", committed_at_ms: 0 } as VersionNode];
    return data.map((c, i) => ({
      id: String(c.version_id),
      position: { x: i * 180, y: 80 + (i % 2) * 40 },
      data: { label: `commit v${c.version_id}\n${c.reason}` },
      style: {
        background: "rgba(17,24,39,0.9)",
        border: "1px solid rgba(148,163,184,0.4)",
        color: "#dbeafe",
        borderRadius: 12,
        width: 150,
        fontSize: 11,
        whiteSpace: "pre-line"
      }
    }));
  }, [commits]);

  const edges = useMemo<Edge[]>(() => {
    const out: Edge[] = [];
    for (let i = 1; i < nodes.length; i++) {
      out.push({ id: `e-${nodes[i - 1].id}-${nodes[i].id}`, source: nodes[i - 1].id, target: nodes[i].id, animated: true });
    }
    return out;
  }, [nodes]);

  return (
    <section className="rounded-2xl border border-slate-700/50 bg-slate-900/50 p-4">
      <div className="mb-3 flex items-center gap-2">
        <GitBranch size={18} className="text-cyan-300" />
        <h2 className="text-sm font-semibold text-slate-100">Autograd Version Graph</h2>
      </div>

      <div className="h-64 rounded-xl border border-slate-700/40 bg-slate-950/50">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          onNodeClick={(_, node) => {
            const c = commits.find((x) => String(x.version_id) === node.id);
            if (c) setSelected(c);
          }}
        >
          <MiniMap />
          <Controls />
        </ReactFlow>
      </div>

      {selected && (
        <div className="mt-3 rounded-xl border border-slate-700/40 bg-slate-950/50 p-3 text-xs text-slate-200">
          <div className="font-semibold text-cyan-200">Version Snapshot v{selected.version_id}</div>
          <div className="mt-1">reason: {selected.reason}</div>
          <div>parent: {selected.parent_version ?? "none"}</div>
          <div>committed_at_ms: {selected.committed_at_ms}</div>
        </div>
      )}
    </section>
  );
}
