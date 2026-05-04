import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import MemorySnapshotPanel from "../components/MemorySnapshotPanel";

describe("MemorySnapshotPanel", () => {
  it("renders selected tensor details including active/shadow ptr", () => {
    render(
      <MemorySnapshotPanel
        tensors={[
          {
            tensor_id: "t0",
            shape: [2, 4],
            strides: [4, 1],
            offset: 0,
            numel: 8,
            dtype: "F32",
            device: "CPU",
            pointers: {
              active_ptr: "arc:active",
              shadow_ptr: "arc:shadow",
              active_version: 1,
              shadow_version: 2
            }
          }
        ]}
      />
    );

    expect(screen.getAllByText("t0")).toHaveLength(2);
    expect(screen.getByText("arc:active")).toBeInTheDocument();
    expect(screen.getByText("arc:shadow")).toBeInTheDocument();
  });

  it("auto-selects first tensor when list appears after mount", () => {
    const { rerender } = render(<MemorySnapshotPanel tensors={[]} />);
    expect(screen.getByText("No tensor selected")).toBeInTheDocument();

    rerender(
      <MemorySnapshotPanel
        tensors={[
          {
            tensor_id: "late",
            shape: [1],
            strides: [1],
            offset: 0,
            numel: 1,
            dtype: "F32",
            device: "CPU",
            pointers: {
              active_ptr: "arc:late",
              shadow_ptr: null,
              active_version: 1,
              shadow_version: null
            }
          }
        ]}
      />
    );

    expect(screen.getByText("arc:late")).toBeInTheDocument();
  });
});
