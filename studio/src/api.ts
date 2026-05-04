import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { EVENTS, EngineStatusDto, EvolutionMetrics, FenceState, HardwareState, VersionNode } from "./types";

export async function getEngineStatus(): Promise<EngineStatusDto> {
  return invoke<EngineStatusDto>("get_engine_status");
}

export async function startEvolutionStream(): Promise<void> {
  return invoke<void>("start_evolution_stream");
}

export async function triggerAtomicSwap(): Promise<void> {
  return invoke<void>("trigger_atomic_swap");
}

export function onMetrics(cb: (m: EvolutionMetrics) => void) {
  return listen<EvolutionMetrics>(EVENTS.METRICS, (e) => cb(e.payload));
}

export function onVersionCommit(cb: (v: VersionNode) => void) {
  return listen<VersionNode>(EVENTS.VERSION_COMMIT, (e) => cb(e.payload));
}

export function onFence(cb: (f: FenceState) => void) {
  return listen<FenceState>(EVENTS.FENCE, (e) => cb(e.payload));
}

export function onHardware(cb: (h: HardwareState) => void) {
  return listen<HardwareState>(EVENTS.HARDWARE, (e) => cb(e.payload));
}
