import { describe, expect, it, vi, beforeEach } from "vitest";
import { EVENTS } from "../types";

const invokeMock = vi.fn();
const listenMock = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => invokeMock(...args)
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: (...args: unknown[]) => listenMock(...args)
}));

import {
  getEngineStatus,
  onFence,
  onHardware,
  onMetrics,
  onVersionCommit,
  startEvolutionStream,
  triggerAtomicSwap
} from "../api";

describe("api bridge", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    listenMock.mockReset();
  });

  it("invokes tauri commands with expected command ids", async () => {
    invokeMock.mockResolvedValue({});

    await getEngineStatus();
    await startEvolutionStream();
    await triggerAtomicSwap();

    expect(invokeMock).toHaveBeenNthCalledWith(1, "get_engine_status");
    expect(invokeMock).toHaveBeenNthCalledWith(2, "start_evolution_stream");
    expect(invokeMock).toHaveBeenNthCalledWith(3, "trigger_atomic_swap");
  });

  it("registers listeners on expected studio event channels", async () => {
    const unlisten = vi.fn();
    listenMock.mockResolvedValue(unlisten);

    await onMetrics(() => undefined);
    await onVersionCommit(() => undefined);
    await onFence(() => undefined);
    await onHardware(() => undefined);

    expect(listenMock).toHaveBeenNthCalledWith(1, EVENTS.METRICS, expect.any(Function));
    expect(listenMock).toHaveBeenNthCalledWith(2, EVENTS.VERSION_COMMIT, expect.any(Function));
    expect(listenMock).toHaveBeenNthCalledWith(3, EVENTS.FENCE, expect.any(Function));
    expect(listenMock).toHaveBeenNthCalledWith(4, EVENTS.HARDWARE, expect.any(Function));
  });
});
