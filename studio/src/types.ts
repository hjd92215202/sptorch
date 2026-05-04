export type UpdatePolicy = "Single" | "Double";

export interface LayerPolicy {
  layer_name: string;
  policy: UpdatePolicy;
}

export interface BufferPointers {
  active_ptr: string;
  shadow_ptr?: string | null;
  active_version: number;
  shadow_version?: number | null;
}

export interface TensorLayoutSnapshot {
  tensor_id: string;
  shape: number[];
  strides: number[];
  offset: number;
  numel: number;
  dtype: string;
  device: string;
  pointers: BufferPointers;
}

export interface VersionNode {
  version_id: number;
  parent_version?: number | null;
  committed_at_ms: number;
  reason: string;
}

export interface EngineStatusDto {
  global_version: number;
  active_version: number;
  chain_head?: VersionNode | null;
  layer_policies: LayerPolicy[];
  tensors: TensorLayoutSnapshot[];
}

export type FencePhase = "Idle" | "Prepare" | "WaitFence" | "Swap" | "Commit" | "Done" | "Error";

export interface FenceState {
  phase: FencePhase;
  progress: number;
  queue_depth: number;
  message: string;
}

export interface EvolutionMetrics {
  ts_ms: number;
  loss: number;
  grad_norm: number;
  grad_scale_factor: number;
  accum_current: number;
  accum_target: number;
  version_id: number;
  fence?: FenceState | null;
}

export interface HardwareState {
  backend: string;
  queue_depth: number;
  online: boolean;
}

export const EVENTS = {
  METRICS: "studio://metrics",
  VERSION_COMMIT: "studio://version-commit",
  FENCE: "studio://fence",
  HARDWARE: "studio://hardware-state"
} as const;
