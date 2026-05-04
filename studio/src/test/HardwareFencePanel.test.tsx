import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import HardwareFencePanel from "../components/HardwareFencePanel";

describe("HardwareFencePanel", () => {
  it("calls onSwap when clicking physical switch button", () => {
    const onSwap = vi.fn();
    render(
      <HardwareFencePanel
        fence={{ phase: "Prepare", progress: 0.2, queue_depth: 8, message: "prepare" }}
        hardware={{ backend: "simulated-hal-ffi", queue_depth: 8, online: true }}
        onSwap={onSwap}
        swapping={false}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /物理切换/i }));
    expect(onSwap).toHaveBeenCalledTimes(1);
  });
});
