import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import VersionedDashboard from "../components/VersionedDashboard";

vi.mock("echarts", () => {
  const setOption = vi.fn();
  const dispose = vi.fn();
  return {
    init: vi.fn(() => ({ setOption, dispose }))
  };
});

describe("VersionedDashboard", () => {
  it("shows accumulation and scale warning when fluctuation exceeds threshold", () => {
    render(
      <VersionedDashboard
        metrics={[
          {
            ts_ms: 1,
            loss: 1.2,
            grad_norm: 0.2,
            grad_scale_factor: 1.0,
            accum_current: 1,
            accum_target: 4,
            version_id: 1,
            fence: null
          },
          {
            ts_ms: 2,
            loss: 1.1,
            grad_norm: 0.3,
            grad_scale_factor: 1.5,
            accum_current: 2,
            accum_target: 4,
            version_id: 1,
            fence: null
          }
        ]}
        commits={[]}
      />
    );

    expect(screen.getByText("2 / 4")).toBeInTheDocument();
    expect(screen.getByText(/scale_gradients 波动/i)).toBeInTheDocument();
  });
});
