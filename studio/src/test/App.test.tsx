import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "../App";
import type { EvolutionMetrics, FenceState, HardwareState, VersionNode } from "../types";

vi.mock("../components/AutogradVersionGraph", () => ({
  default: () => <div data-testid="autograd-graph">autograd-graph</div>
}));

vi.mock("../components/VersionedDashboard", () => ({
  default: ({ metrics, commits }: { metrics: EvolutionMetrics[]; commits: VersionNode[] }) => (
    <div data-testid="versioned-dashboard">
      <div>metrics-count:{metrics.length}</div>
      <div>commits-count:{commits.length}</div>
      <div>latest-loss:{metrics.at(-1)?.loss?.toFixed(4) ?? "-"}</div>
    </div>
  )
}));

vi.mock("../components/MemorySnapshotPanel", () => ({
  default: ({ tensors }: { tensors: unknown[] }) => <div data-testid="memory-panel">tensors-count:{tensors.length}</div>
}));

vi.mock("../components/HardwareFencePanel", () => ({
  default: ({ onSwap, swapping }: { onSwap: () => void; swapping: boolean }) => (
    <button type="button" onClick={onSwap} disabled={swapping}>
      {swapping ? "mocked-swapping" : "mocked-swap"}
    </button>
  )
}));

const getEngineStatusMock = vi.fn();
const startEvolutionStreamMock = vi.fn();
const triggerAtomicSwapMock = vi.fn();

let metricsHandler: ((m: EvolutionMetrics) => void) | null = null;
let commitHandler: ((v: VersionNode) => void) | null = null;
let fenceHandler: ((f: FenceState) => void) | null = null;
let hardwareHandler: ((h: HardwareState) => void) | null = null;

const onMetricsMock = vi.fn(async (cb: (m: EvolutionMetrics) => void) => {
  metricsHandler = cb;
  return () => undefined;
});
const onVersionCommitMock = vi.fn(async (cb: (v: VersionNode) => void) => {
  commitHandler = cb;
  return () => undefined;
});
const onFenceMock = vi.fn(async (cb: (f: FenceState) => void) => {
  fenceHandler = cb;
  return () => undefined;
});
const onHardwareMock = vi.fn(async (cb: (h: HardwareState) => void) => {
  hardwareHandler = cb;
  return () => undefined;
});

vi.mock("../api", () => ({
  getEngineStatus: (...args: unknown[]) => getEngineStatusMock(...args),
  startEvolutionStream: (...args: unknown[]) => startEvolutionStreamMock(...args),
  triggerAtomicSwap: (...args: unknown[]) => triggerAtomicSwapMock(...args),
  onMetrics: (...args: unknown[]) => onMetricsMock(...args),
  onVersionCommit: (...args: unknown[]) => onVersionCommitMock(...args),
  onFence: (...args: unknown[]) => onFenceMock(...args),
  onHardware: (...args: unknown[]) => onHardwareMock(...args)
}));

describe("App event flow", () => {
  beforeEach(() => {
    metricsHandler = null;
    commitHandler = null;
    fenceHandler = null;
    hardwareHandler = null;

    getEngineStatusMock.mockResolvedValue({
      global_version: 1,
      active_version: 1,
      chain_head: {
        version_id: 1,
        parent_version: null,
        committed_at_ms: 1,
        reason: "bootstrap"
      },
      layer_policies: [
        { layer_name: "embedding", policy: "Single" },
        { layer_name: "block0.attn", policy: "Double" }
      ],
      tensors: [
        {
          tensor_id: "t0",
          shape: [2, 4],
          strides: [4, 1],
          offset: 0,
          numel: 8,
          dtype: "F32",
          device: "CPU",
          pointers: { active_ptr: "a", shadow_ptr: null, active_version: 1, shadow_version: null }
        }
      ]
    });
    startEvolutionStreamMock.mockResolvedValue(undefined);
    triggerAtomicSwapMock.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("loads initial status and reacts to metrics/commit/hardware events", async () => {
    render(<App />);

    expect(await screen.findByText("global v1")).toBeInTheDocument();
    expect(screen.getByText("active v1")).toBeInTheDocument();
    expect(screen.getByText("Single Buffer: 1")).toBeInTheDocument();
    expect(screen.getByText("Double Buffer: 1")).toBeInTheDocument();
    expect(screen.getByTestId("memory-panel")).toHaveTextContent("tensors-count:1");

    act(() => {
      metricsHandler?.({
        ts_ms: 10,
        loss: 0.9876,
        grad_norm: 0.2,
        grad_scale_factor: 1.0,
        accum_current: 2,
        accum_target: 4,
        version_id: 1,
        fence: null
      });

      hardwareHandler?.({ backend: "simulated-hal-ffi", queue_depth: 3, online: true });

      commitHandler?.({
        version_id: 2,
        parent_version: 1,
        committed_at_ms: 11,
        reason: "atomic_swap_simulated"
      });
    });

    await waitFor(() => {
      expect(screen.getByText("global v2")).toBeInTheDocument();
      expect(screen.getByText("active v2")).toBeInTheDocument();
      expect(screen.getByTestId("versioned-dashboard")).toHaveTextContent("metrics-count:1");
      expect(screen.getByTestId("versioned-dashboard")).toHaveTextContent("commits-count:2");
      expect(screen.getByText("backend: simulated-hal-ffi")).toBeInTheDocument();
    });
  });

  it("triggers atomic swap command from UI action", async () => {
    render(<App />);
    await screen.findByText("global v1");

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "mocked-swap" }));
    });

    await waitFor(() => {
      expect(triggerAtomicSwapMock).toHaveBeenCalledTimes(1);
    });
  });

  it("keeps latest version on commit and handles done fence signal", async () => {
    render(<App />);
    await screen.findByText("global v1");

    act(() => {
      commitHandler?.({ version_id: 3, parent_version: 1, committed_at_ms: 12, reason: "commit_1" });
      commitHandler?.({ version_id: 4, parent_version: 3, committed_at_ms: 13, reason: "commit_2" });
      fenceHandler?.({ phase: "Done", progress: 1, queue_depth: 0, message: "done" });
    });

    await waitFor(() => {
      expect(screen.getByText("global v4")).toBeInTheDocument();
      expect(screen.getByText("active v4")).toBeInTheDocument();
      expect(screen.getByTestId("versioned-dashboard")).toHaveTextContent("commits-count:3");
    });
  });

  it("deduplicates commit timeline by version_id", async () => {
    render(<App />);
    await screen.findByText("global v1");

    act(() => {
      commitHandler?.({ version_id: 4, parent_version: 1, committed_at_ms: 13, reason: "commit_2" });
      commitHandler?.({ version_id: 4, parent_version: 1, committed_at_ms: 14, reason: "commit_2_retry" });
    });

    await waitFor(() => {
      expect(screen.getByText("global v4")).toBeInTheDocument();
      expect(screen.getByText("active v4")).toBeInTheDocument();
      expect(screen.getByTestId("versioned-dashboard")).toHaveTextContent("commits-count:2");
    });
  });

  it("recovers UI swapping state on fence Error signal", async () => {
    render(<App />);
    await screen.findByText("global v1");

    const btnBefore = screen.getByRole("button", { name: "mocked-swap" });
    expect(btnBefore).toBeEnabled();

    await act(async () => {
      fireEvent.click(btnBefore);
    });

    await waitFor(() => {
      expect(triggerAtomicSwapMock).toHaveBeenCalledTimes(1);
      expect(screen.getByRole("button", { name: "mocked-swapping" })).toBeDisabled();
    });

    act(() => {
      fenceHandler?.({ phase: "Error", progress: 1, queue_depth: 0, message: "swap failed" });
    });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "mocked-swap" })).toBeEnabled();
    });
  });
});
