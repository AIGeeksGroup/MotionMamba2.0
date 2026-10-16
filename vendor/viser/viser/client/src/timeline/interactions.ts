// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type DraggedKeyframe = {
  uuid: string;
  trackId: string;
  initialFrame?: number;
};

export type IntervalDrag = {
  trackId: string;
  startFrame: number;
  endFrame: number;
};

export type DraggedInterval = {
  uuid: string;
  trackId: string;
  originalStartFrame: number;
  originalEndFrame: number;
  clickOffsetFrames: number;
};

export type ResizingInterval = {
  uuid: string;
  trackId: string;
  edge: "left" | "right";
  originalStartFrame: number;
  originalEndFrame: number;
  isKeyframe?: boolean;
  convertedIntervalId?: string;
};

export type ResizingPrompt = {
  uuid: string;
  edge: "left" | "right";
  originalStartFrame: number;
  originalEndFrame: number;
};

export type DraggingIntersection = {
  leftPromptId: string;
  rightPromptId: string;
  originalFrame: number;
  leftOriginalStart: number;
  rightChain: Array<{ uuid: string; originalStart: number; originalEnd: number }>;
  rightChainMinStart: number;
  rightChainMaxEnd: number;
};

export type InteractionState = {
  isDraggingCursor: boolean;
  draggedKeyframe: DraggedKeyframe | null;
  intervalDrag: IntervalDrag | null;
  draggedInterval: DraggedInterval | null;
  resizingInterval: ResizingInterval | null;
  resizingPrompt: ResizingPrompt | null;
  draggingIntersection: DraggingIntersection | null;
};

export const initialInteractionState: InteractionState = {
  isDraggingCursor: false,
  draggedKeyframe: null,
  intervalDrag: null,
  draggedInterval: null,
  resizingInterval: null,
  resizingPrompt: null,
  draggingIntersection: null,
};

export type InteractionAction =
  | { type: "startCursorDrag" }
  | { type: "startKeyframeDrag"; payload: DraggedKeyframe }
  | { type: "startIntervalDrag"; payload: IntervalDrag }
  | { type: "updateIntervalDrag"; payload: IntervalDrag }
  | { type: "startIntervalMove"; payload: DraggedInterval }
  | { type: "startIntervalResize"; payload: ResizingInterval }
  | { type: "updateIntervalResize"; payload: ResizingInterval }
  | { type: "startPromptResize"; payload: ResizingPrompt }
  | { type: "updatePromptResize"; payload: ResizingPrompt }
  | { type: "startIntersectionDrag"; payload: DraggingIntersection }
  | { type: "stopAll" };

export function interactionReducer(
  state: InteractionState,
  action: InteractionAction,
): InteractionState {
  switch (action.type) {
    case "startCursorDrag":
      return { ...initialInteractionState, isDraggingCursor: true };
    case "startKeyframeDrag":
      return { ...initialInteractionState, draggedKeyframe: action.payload };
    case "startIntervalDrag":
      return { ...initialInteractionState, intervalDrag: action.payload };
    case "updateIntervalDrag":
      return { ...state, intervalDrag: action.payload };
    case "startIntervalMove":
      return { ...initialInteractionState, draggedInterval: action.payload };
    case "startIntervalResize":
      return { ...initialInteractionState, resizingInterval: action.payload };
    case "updateIntervalResize":
      return { ...state, resizingInterval: action.payload };
    case "startPromptResize":
      return { ...initialInteractionState, resizingPrompt: action.payload };
    case "updatePromptResize":
      return { ...state, resizingPrompt: action.payload };
    case "startIntersectionDrag":
      return { ...initialInteractionState, draggingIntersection: action.payload };
    case "stopAll":
      return initialInteractionState;
    default:
      return state;
  }
}
