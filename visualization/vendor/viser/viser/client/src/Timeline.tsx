// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  useContext,
} from "react";
import { TimelineMessage } from "./WebsocketMessages";
import { ViewerContext } from "./ViewerContext";
import { FRAME_LABELS_HEIGHT, TRACK_HEIGHT } from "./TimelineConstants";
import {
  DRAW_BUFFER,
  KEYFRAME_HIT_RADIUS,
  KEYFRAME_RADIUS,
  TRACK_LABEL_WIDTH,
} from "./timeline/constants";
import {
  DEFAULT_PROMPT_COLOR,
  HIGHLIGHT_COLOR,
  HIGHLIGHT_BORDER_COLOR,
  HIGHLIGHT_BORDER_WIDTH,
  HIGHLIGHT_SHADOW_BLUR,
  HIGHLIGHT_SHADOW_COLOR,
  HIGHLIGHT_SHADOW_OFFSET_Y,
  LIGHT_THEME,
  DARK_THEME,
  PROMPT_TEXT_SHADOW,
} from "./timeline/styles";
import {
  CanvasSizeState,
  TimelineCoordinates,
  TimelineKeyframe,
  TimelinePrompt,
  TimelineTrack,
} from "./timeline/types";
import {
  drawEndFrameLine,
  drawFrameLabels,
  drawHighlightFrameBox,
  drawInterval,
  drawKeyframe,
  drawLabelOverlay,
} from "./timeline/draw";
import {
  initialInteractionState,
  interactionReducer,
} from "./timeline/interactions";

interface TimelineProps {
  timelineState: TimelineMessage | null;
  onFrameChange?: (frame: number) => void;
  onKeyframeAdd?: (trackId: string, frame: number) => string | void; // Returns the temp UUID
  onKeyframeMove?: (keyframeId: string, newFrame: number) => void;
  onKeyframeDelete?: (keyframeId: string) => void;
  onIntervalAdd?: (trackId: string, startFrame: number, endFrame: number) => string | void; // Returns the temp UUID
  onIntervalMove?: (intervalId: string, newStartFrame: number, newEndFrame: number) => void;
  onIntervalDelete?: (intervalId: string) => void;
  onPromptUpdate?: (promptId: string, newText: string) => void;
  onPromptResize?: (promptId: string, newStartFrame: number, newEndFrame: number) => void;
  onPromptMove?: (promptId: string, newStartFrame: number, newEndFrame: number) => void;
  onPromptDelete?: (promptId: string) => void;
  onPromptAdd?: (startFrame: number, endFrame: number, text: string) => string | void; // Returns the temp UUID
  onPromptSplit?: (promptId: string, splitFrame: number) => void;
  onPromptMerge?: (leftPromptId: string, rightPromptId: string) => void;
  getLatestState?: () => TimelineMessage | null; // Function to get latest state
}

const DEFAULT_KEYFRAME_COLOR: [number, number, number] = [219, 148, 86];
const OUT_OF_RANGE_CONSTRAINT_COLOR: [number, number, number] = [140, 140, 140];
const OUT_OF_RANGE_CONSTRAINT_OPACITY = 0.1;

function getIntervalSegments(
  interval: { start_frame: number; end_frame: number; opacity?: number },
  promptRange: { startFrame: number; endFrame: number } | null,
  trackColor: [number, number, number],
): Array<{ startFrame: number; endFrame: number; color: [number, number, number]; opacity?: number }> {
  const baseOpacity = interval.opacity ?? 1.0;
  const outOfRangeOpacity = Math.min(baseOpacity, OUT_OF_RANGE_CONSTRAINT_OPACITY);
  if (!promptRange) {
    return [
      {
        startFrame: interval.start_frame,
        endFrame: interval.end_frame,
        color: trackColor,
        opacity: baseOpacity,
      },
    ];
  }

  const { startFrame, endFrame } = promptRange;
  if (interval.end_frame < startFrame || interval.start_frame > endFrame) {
    return [
      {
        startFrame: interval.start_frame,
        endFrame: interval.end_frame,
        color: OUT_OF_RANGE_CONSTRAINT_COLOR,
        opacity: outOfRangeOpacity,
      },
    ];
  }

  const segments: Array<{ startFrame: number; endFrame: number; color: [number, number, number]; opacity?: number }> = [];
  const inRangeStart = Math.max(interval.start_frame, startFrame);
  const inRangeEnd = Math.min(interval.end_frame, endFrame);

  if (interval.start_frame < inRangeStart) {
    segments.push({
      startFrame: interval.start_frame,
      endFrame: inRangeStart - 1,
      color: OUT_OF_RANGE_CONSTRAINT_COLOR,
      opacity: outOfRangeOpacity,
    });
  }

  if (inRangeStart <= inRangeEnd) {
    segments.push({
      startFrame: inRangeStart,
      endFrame: inRangeEnd,
      color: trackColor,
      opacity: baseOpacity,
    });
  }

  if (interval.end_frame > inRangeEnd) {
    segments.push({
      startFrame: inRangeEnd + 1,
      endFrame: interval.end_frame,
      color: OUT_OF_RANGE_CONSTRAINT_COLOR,
      opacity: outOfRangeOpacity,
    });
  }

  return segments;
}
// Small baseline tweak so canvas-rendered prompt text matches the edit overlay
// across browsers (native font metrics differ slightly from canvas metrics).
const PROMPT_TEXT_Y_OFFSET_PX = 1;

// ============================================================================
// Helper Types and Functions
// ============================================================================

/**
 * Determine the "zoom envelope" end frame used for coordinate scaling.
 *
 * `end_frame` is often used to set an initial viewport, but `max_frames_zoom`
 * defines how far users should be able to zoom out (number of visible frames).
 */
function getZoomEnvelopeEndFrame(timelineState: TimelineMessage): number {
  const { start_frame, end_frame, max_frames_zoom } = timelineState;
  const maxFramesZoom = Math.max(0, Math.floor(max_frames_zoom ?? 0));
  const envelopeEndByZoom = start_frame + Math.max(0, maxFramesZoom - 1);
  return Math.max(end_frame, envelopeEndByZoom);
}

/**
 * Calculate common coordinate values used throughout timeline rendering
 */
function calculateTimelineCoordinates(
  width: number,
  timelineState: TimelineMessage,
  zoomLevel: number,
  panOffset: number
): TimelineCoordinates {
  const { start_frame } = timelineState;
  const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
  const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
  const visibleFrames = totalFrames / zoomLevel;
  const viewStartFrame = start_frame + panOffset;
  const viewEndFrame = viewStartFrame + visibleFrames;
  
  // Calculate how much space is available for drawing frames (only subtract left padding)
  // Right padding will be enforced by clamping the pan offset
  const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
  const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
  
  const minFramePixels = 60;
  const frameDelta = Math.max(Math.floor(minFramePixels / pixelRange), 1);
  const framesBeforeTimeline = Math.ceil(timelineStartX / pixelRange);
  const firstVisibleFrame = Math.max(0, viewStartFrame - framesBeforeTimeline);
  
  return {
    pixelRange,
    timelineStartX,
    viewStartFrame,
    viewEndFrame,
    visibleFrames,
    minFramePixels,
    frameDelta,
    framesBeforeTimeline,
    firstVisibleFrame,
    endFrame: envelopeEndFrame,
  };
}

// Drawing helpers live in timeline/draw.ts
export function Timeline({ 
  timelineState, 
  onFrameChange,
  onKeyframeAdd,
  onKeyframeMove,
  onKeyframeDelete,
  onIntervalAdd,
  onIntervalMove,
  onIntervalDelete,
  onPromptUpdate,
  onPromptResize,
  onPromptMove,
  onPromptDelete,
  onPromptAdd,
  onPromptSplit,
  onPromptMerge,
  getLatestState
}: TimelineProps) {
  const viewer = useContext(ViewerContext);
  const darkMode = viewer?.useGui((state) => state.theme.dark_mode) ?? true;
  const theme = darkMode ? DARK_THEME : LIGHT_THEME;

  const prompts = useMemo(
    () => timelineState?.prompts ?? [],
    [timelineState?.prompts],
  );
  const promptsByStart = useMemo(
    () => [...prompts].sort((a, b) => a.start_frame - b.start_frame),
    [prompts],
  );
  const promptsByEndDesc = useMemo(
    () => [...prompts].sort((a, b) => b.end_frame - a.end_frame),
    [prompts],
  );
  const promptRange = useMemo(() => {
    if (prompts.length === 0) {
      return null;
    }
    let minStart = prompts[0]!.start_frame;
    let maxEnd = prompts[0]!.end_frame;
    for (const prompt of prompts) {
      minStart = Math.min(minStart, prompt.start_frame);
      maxEnd = Math.max(maxEnd, prompt.end_frame);
    }
    return { startFrame: minStart, endFrame: maxEnd };
  }, [prompts]);
  const tracks = useMemo(
    () => timelineState?.tracks ?? [],
    [timelineState?.tracks],
  );
  const keyframes = useMemo(
    () => timelineState?.keyframes ?? [],
    [timelineState?.keyframes],
  );
  const intervals = useMemo(
    () => timelineState?.intervals ?? [],
    [timelineState?.intervals],
  );
  
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const backgroundCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [editingPrompt, setEditingPrompt] = useState<{uuid: string, text: string, x: number, y: number, width: number, height: number} | null>(null);
  const editingPromptDivRef = useRef<HTMLDivElement | null>(null);
  const [interaction, dispatchInteraction] = useReducer(
    interactionReducer,
    initialInteractionState,
  );
  const {
    isDraggingCursor,
    draggedKeyframe,
    intervalDrag,
    draggedInterval,
    resizingInterval,
    resizingPrompt,
    draggingIntersection,
  } = interaction;
  const [draggedKeyframeFrame, setDraggedKeyframeFrame] = useState<number | null>(null);
  const [draggedIntervalCurrentFrames, setDraggedIntervalCurrentFrames] = useState<{startFrame: number, endFrame: number} | null>(null);
  const [resizingIntervalCurrentFrames, setResizingIntervalCurrentFrames] = useState<{startFrame: number, endFrame: number} | null>(null);
  const [resizingPromptCurrentFrames, setResizingPromptCurrentFrames] = useState<{startFrame: number, endFrame: number} | null>(null);
  const [hoverEdge, setHoverEdge] = useState<'left' | 'right' | null>(null);
  const [hoveredPromptForResize, setHoveredPromptForResize] = useState<{uuid: string, edge: 'left' | 'right', startFrame: number, endFrame: number} | null>(null);
  const [hoveredIntervalForResize, setHoveredIntervalForResize] = useState<{uuid: string, edge: 'left' | 'right', startFrame: number, endFrame: number} | null>(null);
  const [hoveredKeyframe, setHoveredKeyframe] = useState<{uuid: string, frame: number} | null>(null);
  const [hoveredSplitPreview, setHoveredSplitPreview] = useState<{promptId: string, frame: number} | null>(null); // Preview for Shift+hover split
  const [hoveredIntersection, setHoveredIntersection] = useState<{leftPromptId: string, rightPromptId: string, frame: number} | null>(null); // Hover on intersection between prompts
  const [hoveredPromptResize, setHoveredPromptResize] = useState<{promptId: string, frame: number} | null>(null); // Hover on last prompt right-edge resize handle
  const [draggingIntersectionCurrentFrame, setDraggingIntersectionCurrentFrame] = useState<number | null>(null); // Current frame during intersection drag
  const [isHoveringHeader, setIsHoveringHeader] = useState(false);
  const [isHoveringLabelArea, setIsHoveringLabelArea] = useState(false);

  // Clear split preview immediately when Shift is released (no mouse move required).
  useEffect(() => {
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") {
        setHoveredSplitPreview(null);
      }
    };
    const onBlur = () => setHoveredSplitPreview(null);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, []);
  const [resizeRevision, setResizeRevision] = useState(0);
  const [canvasSize, setCanvasSize] = useState<CanvasSizeState>({
    width: 0,
    height: 0,
    dpr: 1,
  });
  
  // Zoom and pan state
  const [zoomLevel, setZoomLevel] = useState(1); // 1 = fit to window, >1 = zoomed in
  const [panOffset, setPanOffset] = useState(0); // Offset in frames
  const didInitZoomRef = useRef(false);
  const lastInitZoomConfigRef = useRef<string | null>(null);
  const didUserAdjustViewRef = useRef(false);
  const lastZoomDefaultsRef = useRef<string | null>(null);

  // Keep latest values available to requestAnimationFrame loops without stale closures.
  const timelineStateRef = useRef<typeof timelineState | null>(null);
  const zoomLevelRef = useRef<number>(zoomLevel);
  const panOffsetRef = useRef<number>(panOffset);
  const resizingPromptRef = useRef<typeof resizingPrompt | null>(null);
  const resizingPromptCurrentFramesRef = useRef<typeof resizingPromptCurrentFrames | null>(null);
  const draggingIntersectionRef = useRef<typeof draggingIntersection | null>(null);
  const onPromptResizeRef = useRef<typeof onPromptResize | null>(null);
  const onPromptMoveRef = useRef<typeof onPromptMove | null>(null);

  useEffect(() => {
    timelineStateRef.current = timelineState;
  }, [timelineState]);
  useEffect(() => {
    zoomLevelRef.current = zoomLevel;
  }, [zoomLevel]);
  useEffect(() => {
    panOffsetRef.current = panOffset;
  }, [panOffset]);
  useEffect(() => {
    resizingPromptRef.current = resizingPrompt;
  }, [resizingPrompt]);
  useEffect(() => {
    resizingPromptCurrentFramesRef.current = resizingPromptCurrentFrames;
  }, [resizingPromptCurrentFrames]);
  useEffect(() => {
    draggingIntersectionRef.current = draggingIntersection;
  }, [draggingIntersection]);
  useEffect(() => {
    onPromptResizeRef.current = onPromptResize ?? null;
  }, [onPromptResize]);
  useEffect(() => {
    onPromptMoveRef.current = onPromptMove ?? null;
  }, [onPromptMove]);

  // Auto-pan (aka "autoscroll") while resizing the RIGHT edge of a prompt, so the user can keep dragging.
  const AUTO_PAN_EDGE_PX = 60;
  const AUTO_PAN_MAX_PX_PER_SEC = 900;
  type AutoPanMode = "resizeRightEdge" | "intersection";
  const autoPanModeRef = useRef<AutoPanMode | null>(null);
  const autoPanRafRef = useRef<number | null>(null);
  const autoPanLastTsRef = useRef<number | null>(null);
  const autoPanCanvasXRef = useRef<number | null>(null);

  const stopAutoPan = useCallback(() => {
    if (autoPanRafRef.current !== null) {
      cancelAnimationFrame(autoPanRafRef.current);
      autoPanRafRef.current = null;
    }
    autoPanLastTsRef.current = null;
    autoPanModeRef.current = null;
  }, []);

  const getFrameFromXWithPan = useCallback((x: number, panOffsetOverride: number): number => {
    const ts = timelineStateRef.current;
    const canvas = canvasRef.current;
    if (!ts || !canvas) return 0;

    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const { start_frame } = ts;

    const envelopeEndFrame = getZoomEnvelopeEndFrame(ts);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevelRef.current;
    const viewStartFrame = start_frame + panOffsetOverride;

    const pixelRange = (width - TRACK_LABEL_WIDTH - 2 * KEYFRAME_RADIUS) / visibleFrames;
    const adjustedX = x - TRACK_LABEL_WIDTH - 2 * KEYFRAME_RADIUS;
    const frame = Math.round(adjustedX / pixelRange + viewStartFrame);
    return Math.max(start_frame, frame);
  }, []);

  const getPromptRightEdgeBounds = useCallback((
    ts: NonNullable<typeof timelineStateRef.current>,
    rp: NonNullable<typeof resizingPromptRef.current>,
  ) => {
    const PROMPT_MIN_FRAMES = ts.min_prompt_duration || 1;
    const PROMPT_MAX_FRAMES = ts.max_prompt_duration || Infinity;
    const fixedStart = rp.originalStartFrame;
    const fixedEnd = rp.originalEndFrame;

    const { prompts } = ts;
    let rightNeighborStart = Infinity;
    for (const p of prompts) {
      if (p.uuid === rp.uuid) continue;
      if (p.start_frame >= fixedEnd) {
        rightNeighborStart = Math.min(rightNeighborStart, p.start_frame);
      }
    }

    const minAllowedEnd = fixedStart + PROMPT_MIN_FRAMES;
    const maxEndFromMax = Number.isFinite(PROMPT_MAX_FRAMES) ? (fixedStart + PROMPT_MAX_FRAMES) : Infinity;
    const maxAllowedEnd = Math.min(maxEndFromMax, rightNeighborStart);
    return { minAllowedEnd, maxAllowedEnd };
  }, []);

  const applyResizingPromptAtFrame = useCallback((frame: number) => {
    const rp = resizingPromptRef.current;
    const ts = timelineStateRef.current;
    if (!rp || !ts) return;

    const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
    const PROMPT_MIN_FRAMES = ts.min_prompt_duration || 1;
    const PROMPT_MAX_FRAMES = ts.max_prompt_duration || Infinity;

    const fixedStart = rp.originalStartFrame;
    const fixedEnd = rp.originalEndFrame;

    const { start_frame, prompts } = ts;

    let leftNeighborEnd = start_frame;
    let rightNeighborStart = Infinity;
    for (const p of prompts) {
      if (p.uuid === rp.uuid) continue;
      if (p.end_frame <= fixedStart) {
        leftNeighborEnd = Math.max(leftNeighborEnd, p.end_frame);
      }
      if (p.start_frame >= fixedEnd) {
        rightNeighborStart = Math.min(rightNeighborStart, p.start_frame);
      }
    }

    let validStartFrame = fixedStart;
    let validEndFrame = fixedEnd;

    if (rp.edge === "left") {
      const maxStartFromMin = fixedEnd - PROMPT_MIN_FRAMES;
      const minStartFromMax = Number.isFinite(PROMPT_MAX_FRAMES) ? (fixedEnd - PROMPT_MAX_FRAMES) : -Infinity;
      const minAllowedStart = Math.max(leftNeighborEnd, minStartFromMax);
      const maxAllowedStart = maxStartFromMin;

      if (maxAllowedStart < minAllowedStart) {
        validStartFrame = minAllowedStart;
      } else {
        validStartFrame = clamp(frame, minAllowedStart, maxAllowedStart);
      }
    } else {
      const minEndFromMin = fixedStart + PROMPT_MIN_FRAMES;
      const maxEndFromMax = Number.isFinite(PROMPT_MAX_FRAMES) ? (fixedStart + PROMPT_MAX_FRAMES) : Infinity;
      const minAllowedEnd = minEndFromMin;
      const maxAllowedEnd = Math.min(maxEndFromMax, rightNeighborStart);

      if (maxAllowedEnd < minAllowedEnd) {
        validEndFrame = maxAllowedEnd;
      } else {
        validEndFrame = clamp(frame, minAllowedEnd, maxAllowedEnd);
      }
    }

    setResizingPromptCurrentFrames({ startFrame: validStartFrame, endFrame: validEndFrame });
    const cb = onPromptResizeRef.current;
    if (cb) {
      cb(rp.uuid, validStartFrame, validEndFrame);
    }
  }, []);

  const applyDraggingIntersectionAtFrame = useCallback((frame: number) => {
    const di = draggingIntersectionRef.current;
    const ts = timelineStateRef.current;
    if (!di || !ts) return;

    const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
    const PROMPT_MIN_FRAMES = ts.min_prompt_duration || 1;
    const PROMPT_MAX_FRAMES = ts.max_prompt_duration || Infinity;

    const minFrameFromLeft = di.leftOriginalStart + PROMPT_MIN_FRAMES;
    const maxFrameFromLeft = di.leftOriginalStart + PROMPT_MAX_FRAMES;

    const timelineStartFrame = ts.start_frame;
    const deltaMin = timelineStartFrame - di.rightChainMinStart;
    const minFrameFromChain = di.originalFrame + deltaMin;

    const minFrame = Math.max(minFrameFromLeft, minFrameFromChain);
    const maxFrame = maxFrameFromLeft;

    const constrainedFrame = clamp(frame, minFrame, maxFrame);
    const delta = constrainedFrame - di.originalFrame;

    setDraggingIntersectionCurrentFrame(constrainedFrame);

    onPromptResizeRef.current?.(di.leftPromptId, di.leftOriginalStart, constrainedFrame);
    const move = onPromptMoveRef.current;
    if (move) {
      for (const p of di.rightChain) {
        move(p.uuid, p.originalStart + delta, p.originalEnd + delta);
      }
    }
  }, []);

  const tickAutoPan = useCallback((tsNow: number) => {
    const mode = autoPanModeRef.current;
    const rp = resizingPromptRef.current;
    const di = draggingIntersectionRef.current;
    const ts = timelineStateRef.current;
    const canvas = canvasRef.current;
    const cursorX = autoPanCanvasXRef.current;

    const isResizeRightActive = !!rp && rp.edge === "right";
    const isIntersectionActive = !!di;
    const modeActive =
      (mode === "resizeRightEdge" && isResizeRightActive) ||
      (mode === "intersection" && isIntersectionActive);

    if (!modeActive || !ts || !canvas || cursorX === null) {
      stopAutoPan();
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const timelineStartX = TRACK_LABEL_WIDTH + 2 * KEYFRAME_RADIUS;
    const leftThreshold = timelineStartX + AUTO_PAN_EDGE_PX;
    const rightThreshold = width - AUTO_PAN_EDGE_PX;

    // Positive => pan right, Negative => pan left.
    const rightOver = cursorX - rightThreshold;
    const leftOver = leftThreshold - cursorX;
    const intensityRight = Math.max(0, Math.min(2, rightOver / AUTO_PAN_EDGE_PX)); // allow a bit of extra speed if beyond edge
    const intensityLeft = Math.max(0, Math.min(2, leftOver / AUTO_PAN_EDGE_PX));
    const signedIntensity = intensityRight > 0 ? intensityRight : (intensityLeft > 0 ? -intensityLeft : 0);

    if (signedIntensity === 0) {
      stopAutoPan();
      return;
    }

    const last = autoPanLastTsRef.current ?? tsNow;
    const dtMs = Math.max(0, tsNow - last);
    autoPanLastTsRef.current = tsNow;

    const { start_frame } = ts;
    const envelopeEndFrame = getZoomEnvelopeEndFrame(ts);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevelRef.current;
    const pixelRange = (width - TRACK_LABEL_WIDTH - 2 * KEYFRAME_RADIUS) / visibleFrames; // px per frame
    const framesPerPx = pixelRange > 0 ? 1 / pixelRange : 0;

    // If the dragged value is already clamped by constraints, don't keep panning in the blocked direction.
    if (mode === "resizeRightEdge" && rp) {
      const { minAllowedEnd, maxAllowedEnd } = getPromptRightEdgeBounds(ts, rp);
      const currentEnd = resizingPromptCurrentFramesRef.current?.endFrame ?? rp.originalEndFrame;

      if (!Number.isFinite(minAllowedEnd) || !Number.isFinite(maxAllowedEnd) || maxAllowedEnd <= minAllowedEnd) {
        stopAutoPan();
        return;
      }
      if (signedIntensity > 0 && currentEnd >= maxAllowedEnd) {
        stopAutoPan();
        return;
      }
      if (signedIntensity < 0 && currentEnd <= minAllowedEnd) {
        stopAutoPan();
        return;
      }
    }

    if (mode === "intersection" && di) {
      const PROMPT_MIN_FRAMES = ts.min_prompt_duration || 1;
      const PROMPT_MAX_FRAMES = ts.max_prompt_duration || Infinity;

      const minFrameFromLeft = di.leftOriginalStart + PROMPT_MIN_FRAMES;
      const maxFrameFromLeft = di.leftOriginalStart + PROMPT_MAX_FRAMES;
      const timelineStartFrame = ts.start_frame;
      const deltaMin = timelineStartFrame - di.rightChainMinStart;
      const minFrameFromChain = di.originalFrame + deltaMin;
      const minFrame = Math.max(minFrameFromLeft, minFrameFromChain);
      const maxFrame = maxFrameFromLeft;

      const currentFrame = getFrameFromXWithPan(cursorX, panOffsetRef.current);
      const constrainedFrame = Math.max(minFrame, Math.min(maxFrame, currentFrame));
      if (signedIntensity > 0 && constrainedFrame >= maxFrame) {
        stopAutoPan();
        return;
      }
      if (signedIntensity < 0 && constrainedFrame <= minFrame) {
        stopAutoPan();
        return;
      }
    }

    const deltaPx = (AUTO_PAN_MAX_PX_PER_SEC * Math.abs(signedIntensity)) * (dtMs / 1000);
    const deltaFrames = deltaPx * framesPerPx * (signedIntensity > 0 ? 1 : -1);

    if (deltaFrames !== 0) {
      const prevPan = panOffsetRef.current;
      const nextPan = Math.max(0, prevPan + deltaFrames);
      if (nextPan === prevPan) {
        stopAutoPan();
        return;
      }

      didUserAdjustViewRef.current = true;
      panOffsetRef.current = nextPan;
      setPanOffset(nextPan);

      const frame = getFrameFromXWithPan(cursorX, nextPan);
      if (mode === "resizeRightEdge") {
        applyResizingPromptAtFrame(frame);
      } else if (mode === "intersection") {
        applyDraggingIntersectionAtFrame(frame);
      }
    }

    autoPanRafRef.current = requestAnimationFrame(tickAutoPan);
  }, [applyDraggingIntersectionAtFrame, applyResizingPromptAtFrame, getFrameFromXWithPan, getPromptRightEdgeBounds, stopAutoPan]);

  const startAutoPan = useCallback((mode: AutoPanMode) => {
    autoPanModeRef.current = mode;
    // Reset timestamp so switching modes or re-entering edge doesn't cause a big dt jump.
    autoPanLastTsRef.current = null;
    if (autoPanRafRef.current !== null) return;
    autoPanRafRef.current = requestAnimationFrame(tickAutoPan);
  }, [tickAutoPan]);

  // Ensure we never leave a RAF loop running after resize ends / component unmounts.
  useEffect(() => {
    const hasResizeRight = !!resizingPrompt && resizingPrompt.edge === "right";
    const hasIntersection = !!draggingIntersection;
    if (!hasResizeRight && !hasIntersection) {
      stopAutoPan();
    }
    return () => stopAutoPan();
  }, [resizingPrompt, draggingIntersection, stopAutoPan]);

  // Initialize zoom so we start by showing [start_frame, start_frame + default_num_frames_zoom].
  useEffect(() => {
    if (!timelineState) return;
    const { start_frame, end_frame, default_num_frames_zoom, max_frames_zoom } = timelineState;
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);

    const configKey = `${start_frame}|${end_frame}|${default_num_frames_zoom ?? ""}|${max_frames_zoom ?? ""}`;
    const configUnchanged = lastInitZoomConfigRef.current === configKey;
    const zoomDefaultsKey = `${default_num_frames_zoom ?? ""}|${max_frames_zoom ?? ""}`;
    const zoomDefaultsChanged = lastZoomDefaultsRef.current !== null && lastZoomDefaultsRef.current !== zoomDefaultsKey;

    // Don't override the user's view once they've manually panned/zoomed.
    // But if the user hasn't interacted yet, allow updates (e.g., a later Python `set_defaults()`)
    // to re-initialize zoom.
    //
    // Exception: if the server explicitly changes the zoom defaults (e.g. loading a new motion/example),
    // we do want to re-initialize so the viewport adapts.
    if (didInitZoomRef.current && ((didUserAdjustViewRef.current && !zoomDefaultsChanged) || configUnchanged)) {
      return;
    }

    const initialVisible = Math.max(
      1,
      Math.min(default_num_frames_zoom ?? 300, max_frames_zoom ?? 1000, totalFrames),
    );

    // Previously this was hard-clamped to 20, which prevents `default_num_frames_zoom`
    // from taking effect for large ranges. Allow zoom up to "1 visible frame" if desired.
    const maxZoomLevel = Math.max(20, totalFrames);
    const newZoomLevel = Math.max(1, Math.min(maxZoomLevel, totalFrames / initialVisible));
    setZoomLevel(newZoomLevel);
    setPanOffset(0);
    didInitZoomRef.current = true;
    lastInitZoomConfigRef.current = configKey;
    lastZoomDefaultsRef.current = zoomDefaultsKey;
  }, [
    timelineState?.start_frame,
    timelineState?.end_frame,
    timelineState?.default_num_frames_zoom,
    timelineState?.max_frames_zoom,
  ]);

  // Keep server-driven frame changes visible without disturbing a viewport
  // that already contains the cursor. This covers GUI frame inputs, playback,
  // and other non-canvas seeks while preserving manual pan/zoom until the
  // cursor actually leaves the visible window.
  useEffect(() => {
    if (!timelineState || !didInitZoomRef.current) return;
    const totalFrames = Math.max(
      1,
      getZoomEnvelopeEndFrame(timelineState) - timelineState.start_frame + 1,
    );
    const visibleFrames = totalFrames / zoomLevel;
    const viewStartFrame = timelineState.start_frame + panOffset;
    const viewEndFrame = viewStartFrame + visibleFrames;
    const currentFrame = timelineState.current_frame;
    if (currentFrame >= viewStartFrame && currentFrame <= viewEndFrame) return;

    const centeredStart = currentFrame - visibleFrames / 2;
    const nextPanOffset = Math.max(0, centeredStart - timelineState.start_frame);
    panOffsetRef.current = nextPanOffset;
    setPanOffset(nextPanOffset);
  }, [timelineState?.current_frame, timelineState?.start_frame, zoomLevel, panOffset]);
  const scrollModeRef = useRef<'zoom' | 'pan' | null>(null); // Lock scroll mode during gesture
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null); // Detect when scrolling ends
  const lastScrollTimeRef = useRef<number>(0); // Track time of last scroll event
  const lastScrollDeltaRef = useRef<number>(0); // Track delta of last scroll event for velocity
  const scrollVelocityRef = useRef<number>(0); // Track scroll velocity to detect new gestures

  // Clear hover states when prompts change (e.g., when a prompt is deleted)
  useEffect(() => {
    if (!timelineState) return;
    
    // If we're hovering over an intersection, check if it still exists
    if (hoveredIntersection) {
      const leftPromptExists = timelineState.prompts?.some(p => p.uuid === hoveredIntersection.leftPromptId);
      const rightPromptExists = timelineState.prompts?.some(p => p.uuid === hoveredIntersection.rightPromptId);
      
      // If either prompt no longer exists, clear the hover state
      if (!leftPromptExists || !rightPromptExists) {
        setHoveredIntersection(null);
      }
    }
  }, [timelineState?.prompts, hoveredIntersection]);

  // Calculate frame from mouse position
  const getFrameFromX = useCallback(
    (x: number): number => {
      if (!timelineState || !canvasRef.current) return 0;
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const { start_frame } = timelineState;
      
      // Calculate visible frame range based on zoom and pan
      const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
      const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
      const visibleFrames = totalFrames / zoomLevel;
      const viewStartFrame = start_frame + panOffset;
      
      const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
      const adjustedX = x - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS;
      const frame = Math.round(adjustedX / pixelRange + viewStartFrame);
      // Only clamp to start_frame (allow infinite scrolling to the right)
      return Math.max(start_frame, frame);
    },
    [timelineState, zoomLevel, panOffset],
  );

  // Helper: Get track index from Y position
  // Helper: Calculate track Y position by accumulating heights
  const getTrackY = useCallback((trackIndex: number): number => {
    if (!timelineState) return FRAME_LABELS_HEIGHT;
    
    let y = FRAME_LABELS_HEIGHT;
    
    // Add prompt track height (always shown)
    y += TRACK_HEIGHT;
    
    // Add heights of all tracks before this one
    for (let i = 0; i < trackIndex; i++) {
      y += TRACK_HEIGHT * (timelineState.tracks[i].height_scale || 1.0);
    }
    
    return y;
  }, [timelineState, zoomLevel, panOffset]);

  const getTrackFromY = useCallback((y: number): { type: 'header' | 'prompt' | 'track', trackIndex?: number } => {
    if (!timelineState) return { type: 'header' };
    
    if (y < FRAME_LABELS_HEIGHT) {
      return { type: 'header' };
    }
    
    let currentY = FRAME_LABELS_HEIGHT;
    
    // Check if in prompt track (always shown)
    if (y >= currentY && y < currentY + TRACK_HEIGHT) {
      return { type: 'prompt' };
    }
    currentY += TRACK_HEIGHT;
    
    // Check each track by accumulating heights
    for (let i = 0; i < timelineState.tracks.length; i++) {
      const trackHeight = TRACK_HEIGHT * (timelineState.tracks[i].height_scale || 1.0);
      if (y >= currentY && y < currentY + trackHeight) {
        return { type: 'track', trackIndex: i };
      }
      currentY += trackHeight;
    }
    
    return { type: 'header' };
  }, [timelineState, zoomLevel, panOffset]);

  // Helper: Find keyframe at position
  const findKeyframeAtPosition = useCallback((x: number, y: number): TimelineKeyframe | null => {
    if (!timelineState || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const { start_frame, keyframes } = timelineState;
    
    // Calculate visible frame range based on zoom and pan
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevel;
    const viewStartFrame = start_frame + panOffset;
    
    const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
    const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
    
    const trackInfo = getTrackFromY(y);
    if (trackInfo.type !== 'track' || trackInfo.trackIndex === undefined) return null;
    
    const track = timelineState.tracks[trackInfo.trackIndex];
    const trackY = getTrackY(trackInfo.trackIndex);
    const trackHeight = TRACK_HEIGHT * (track.height_scale || 1.0);
    const keyframeY = trackY + trackHeight / 2;
    
    // Check if click is near a keyframe
    for (const kf of keyframes) {
      // Skip locked keyframes - they cannot be moved or deleted
      if (kf.locked) continue;
      
      if (kf.track_id === track.uuid) {
        const kfX = timelineStartX + (kf.frame - viewStartFrame) * pixelRange;
        const distance = Math.sqrt(Math.pow(x - kfX, 2) + Math.pow(y - keyframeY, 2));
        if (distance <= KEYFRAME_HIT_RADIUS) {
          return kf;
        }
      }
    }
    
    return null;
  }, [timelineState, getTrackFromY, getTrackY]);

  const findIntervalAtPosition = useCallback((x: number, y: number): {interval: any, clickOffsetFrames: number} | null => {
    if (!timelineState || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const { start_frame } = timelineState;
    
    // Calculate visible frame range based on zoom and pan
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevel;
    const viewStartFrame = start_frame + panOffset;
    
    const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
    const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
    
    const trackInfo = getTrackFromY(y);
    if (trackInfo.type !== 'track' || trackInfo.trackIndex === undefined) return null;
    
    const track = timelineState.tracks[trackInfo.trackIndex];
    const trackY = getTrackY(trackInfo.trackIndex);
    const trackHeight = TRACK_HEIGHT * (track.height_scale || 1.0);
    
    // Check intervals for this track (check in reverse order so topmost/latest intervals are hit first)
    for (let i = intervals.length - 1; i >= 0; i--) {
      const interval = intervals[i];
      
      // Skip locked intervals - they cannot be moved or deleted
      if (interval.locked) continue;
      
      if (interval.track_id === track.uuid) {
        const intervalFrameWidth = interval.end_frame - interval.start_frame + 1;
        
        // Position based on start frame (aligned with keyframes)
        const frameStartX = timelineStartX + (interval.start_frame - viewStartFrame) * pixelRange;
        
        // For single-frame intervals (circles), use circular hit detection
        if (intervalFrameWidth === 1) {
          const centerY = trackY + trackHeight / 2;
          const distance = Math.sqrt(Math.pow(x - frameStartX, 2) + Math.pow(y - centerY, 2));
          
          if (distance <= KEYFRAME_RADIUS) {
            return { interval, clickOffsetFrames: 0 }; // Offset is 0 for circles
          }
        } else {
          // For multi-frame intervals, use rectangle hit detection
          // Calculate width same as prompts (inclusive)
          const intervalWidth = (interval.end_frame - interval.start_frame) * pixelRange + 2 * KEYFRAME_RADIUS;
          const startX = frameStartX - KEYFRAME_RADIUS;
          const endX = startX + intervalWidth;
          const intervalHeight = KEYFRAME_RADIUS * 2;
          const intervalY = trackY + (trackHeight - intervalHeight) / 2;
          
          // Check if click is within the interval rectangle
          if (x >= startX && x <= endX && y >= intervalY && y <= intervalY + intervalHeight) {
            const clickFrame = getFrameFromX(x);
            const clickOffsetFrames = clickFrame - interval.start_frame;
            return { interval, clickOffsetFrames };
          }
        }
      }
    }
    
    return null;
  }, [timelineState, intervals, getTrackFromY, getTrackY, getFrameFromX]);

  // Helper: Find prompt at position (for editing)
  const findPromptAtPosition = useCallback((x: number, y: number): TimelinePrompt | null => {
    if (!timelineState || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const { start_frame } = timelineState;
    
    // Calculate visible frame range based on zoom and pan
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevel;
    const viewStartFrame = start_frame + panOffset;
    
    const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
    const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
    
    // Check if we're in the prompts track
    if (prompts.length === 0) return null;
    if (y < FRAME_LABELS_HEIGHT || y >= FRAME_LABELS_HEIGHT + TRACK_HEIGHT) return null;
    
    // Check each prompt
    for (const prompt of prompts) {
      const startX = timelineStartX + (prompt.start_frame - viewStartFrame) * pixelRange;
      const endX = timelineStartX + (prompt.end_frame - viewStartFrame) * pixelRange;
      const promptY = FRAME_LABELS_HEIGHT + 1;
      const promptHeight = TRACK_HEIGHT - 2;
      
      if (x >= startX && x <= endX && y >= promptY && y <= promptY + promptHeight) {
        return prompt;
      }
    }
    
    return null;
  }, [timelineState, zoomLevel, panOffset]);

  // Helper: Find if mouse is near a prompt edge (for resizing)
  const findPromptEdge = useCallback((x: number, y: number): {prompt: TimelinePrompt, edge: 'left' | 'right'} | null => {
    if (!timelineState || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const { start_frame } = timelineState;
    
    // Calculate visible frame range based on zoom and pan
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevel;
    const viewStartFrame = start_frame + panOffset;
    
    const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
    const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
    const edgeThreshold = 10; // pixels from edge to trigger resize (match intersection hitbox)
    
    // Check if we're in the prompts track
    if (prompts.length === 0) return null;
    if (y < FRAME_LABELS_HEIGHT || y >= FRAME_LABELS_HEIGHT + TRACK_HEIGHT) return null;
    
    const promptY = FRAME_LABELS_HEIGHT + 2;
    const promptHeight = TRACK_HEIGHT - 4;
    
    // Check each prompt's edges
    const candidates: Array<{prompt: TimelinePrompt, edge: 'left' | 'right', edgeX: number, distance: number}> = [];
    
    for (const prompt of prompts) {
      const startX = timelineStartX + (prompt.start_frame - viewStartFrame) * pixelRange;
      const endX = timelineStartX + (prompt.end_frame - viewStartFrame) * pixelRange;
      
      if (y >= promptY && y <= promptY + promptHeight) {
        // Check left edge
        if (Math.abs(x - startX) <= edgeThreshold) {
          candidates.push({ 
            prompt, 
            edge: 'left', 
            edgeX: startX,
            distance: Math.abs(x - startX)
          });
        }
        // Check right edge
        if (Math.abs(x - endX) <= edgeThreshold) {
          candidates.push({ 
            prompt, 
            edge: 'right', 
            edgeX: endX,
            distance: Math.abs(x - endX)
          });
        }
      }
    }
    
    // If multiple edges at the same position (prompts touching), pick based on which side of the boundary
    if (candidates.length > 0) {
      // Sort by distance to mouse
      candidates.sort((a, b) => a.distance - b.distance);
      
      // If there are multiple candidates at the same position (or very close), disambiguate
      const closest = candidates[0];
      const samePosition = candidates.filter(c => Math.abs(c.edgeX - closest.edgeX) < 1);
      
      if (samePosition.length > 1) {
        // Multiple edges at same position - pick based on which side of boundary
        // If mouse is to the left of the boundary, prefer right edge (resizing left prompt)
        // If mouse is to the right of the boundary, prefer left edge (resizing right prompt)
        const boundaryX = closest.edgeX;
        
        if (x < boundaryX) {
          // Mouse is left of boundary - prefer right edge (end of the left prompt)
          const rightEdge = samePosition.find(c => c.edge === 'right');
          if (rightEdge) return { prompt: rightEdge.prompt, edge: rightEdge.edge };
        } else {
          // Mouse is right of boundary - prefer left edge (start of the right prompt)
          const leftEdge = samePosition.find(c => c.edge === 'left');
          if (leftEdge) return { prompt: leftEdge.prompt, edge: leftEdge.edge };
        }
      }
      
      // Return the closest one
      return { prompt: closest.prompt, edge: closest.edge };
    }
    
    return null;
  }, [timelineState, zoomLevel, panOffset]);

  // Helper: Find if mouse is near a prompt intersection (between two adjacent prompts)
  const findPromptIntersection = useCallback((x: number, y: number): {leftPrompt: TimelinePrompt, rightPrompt: TimelinePrompt, frame: number} | null => {
    if (!timelineState || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const { start_frame } = timelineState;
    
    if (promptsByStart.length < 2) return null;
    
    // Calculate visible frame range based on zoom and pan
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevel;
    const viewStartFrame = start_frame + panOffset;
    
    const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
    const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
    const intersectionThreshold = 10; // pixels from intersection point to trigger
    
    // Check if we're in the prompt track area
    const trackInfo = getTrackFromY(y);
    if (trackInfo.type !== 'prompt') return null;
    
    // Sort prompts by start frame
    // Find adjacent prompts that share a boundary
    for (let i = 0; i < promptsByStart.length - 1; i++) {
      const leftPrompt = promptsByStart[i];
      const rightPrompt = promptsByStart[i + 1];
      
      // Check if they are adjacent (share the same boundary frame)
      if (leftPrompt.end_frame === rightPrompt.start_frame) {
        const intersectionFrame = leftPrompt.end_frame;
        const intersectionX = timelineStartX + (intersectionFrame - viewStartFrame) * pixelRange;
        
        // Check if mouse is near this intersection
        if (Math.abs(x - intersectionX) <= intersectionThreshold) {
          return {
            leftPrompt: leftPrompt,
            rightPrompt: rightPrompt,
            frame: intersectionFrame
          };
        }
      }
    }
    
    return null;
  }, [timelineState, promptsByStart, zoomLevel, panOffset, getTrackFromY]);

  // Helper: Find if mouse is near an interval edge (for resizing)
  const findIntervalEdge = useCallback((x: number, y: number): {interval: any, edge: 'left' | 'right'} | null => {
    if (!timelineState || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const { start_frame } = timelineState;
    
    // Calculate visible frame range based on zoom and pan
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    const visibleFrames = totalFrames / zoomLevel;
    const viewStartFrame = start_frame + panOffset;
    
    const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
    const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
    const edgeThreshold = 8; // pixels from edge to trigger resize
    
    const trackInfo = getTrackFromY(y);
    if (trackInfo.type !== 'track' || trackInfo.trackIndex === undefined) return null;
    
    const track = timelineState.tracks[trackInfo.trackIndex];
    const trackY = getTrackY(trackInfo.trackIndex);
    const trackHeight = TRACK_HEIGHT * (track.height_scale || 1.0);
    
    // Check intervals first
    for (let i = intervals.length - 1; i >= 0; i--) {
      const interval = intervals[i];
      
      // Skip locked intervals - they cannot be resized
      if (interval.locked) continue;
      
      if (interval.track_id === track.uuid) {
        const intervalFrameWidth = interval.end_frame - interval.start_frame + 1;
        
        // Position based on start frame (aligned with keyframes)
        const frameStartX = timelineStartX + (interval.start_frame - viewStartFrame) * pixelRange;
        
        // For single-frame intervals (circles), use circular hit detection
        if (intervalFrameWidth === 1) {
          const centerY = trackY + trackHeight / 2;
          const distance = Math.sqrt(Math.pow(x - frameStartX, 2) + Math.pow(y - centerY, 2));
          
          if (distance <= KEYFRAME_RADIUS + edgeThreshold) {
            // Determine which side based on x position relative to center
            if (x < frameStartX) {
              return { interval, edge: 'left' };
            } else {
              return { interval, edge: 'right' };
            }
          }
        } else {
          // For multi-frame intervals, check edges
          // Calculate width same as prompts (inclusive)
          const intervalWidth = (interval.end_frame - interval.start_frame) * pixelRange + 2 * KEYFRAME_RADIUS;
          const startX = frameStartX - KEYFRAME_RADIUS;
          const endX = startX + intervalWidth;
          
          const intervalHeight = KEYFRAME_RADIUS * 2;
          const intervalY = trackY + (trackHeight - intervalHeight) / 2;
          
          // Check if near edges
          if (y >= intervalY && y <= intervalY + intervalHeight) {
            if (Math.abs(x - startX) <= edgeThreshold && x >= startX - edgeThreshold && x <= startX + edgeThreshold) {
              return { interval, edge: 'left' };
            }
            if (Math.abs(x - endX) <= edgeThreshold && x >= endX - edgeThreshold && x <= endX + edgeThreshold) {
              return { interval, edge: 'right' };
            }
          }
        }
      }
    }
    
    // Check single keyframes (to convert to intervals)
    for (const kf of keyframes) {
      // Skip locked keyframes - they cannot be converted to intervals
      if (kf.locked) continue;
      
      if (kf.track_id === track.uuid) {
        const kfX = timelineStartX + (kf.frame - viewStartFrame) * pixelRange;
        const kfY = trackY + trackHeight / 2;
        const distance = Math.sqrt(Math.pow(x - kfX, 2) + Math.pow(y - kfY, 2));
        
        if (distance <= KEYFRAME_RADIUS + edgeThreshold) {
          // Treat single keyframe as interval with start=end=frame
          const fakeInterval = {
            uuid: kf.uuid,
            track_id: kf.track_id,
            start_frame: kf.frame,
            end_frame: kf.frame,
            value: kf.value
          };
          // Determine which side based on x position
          if (x < kfX) {
            return { interval: fakeInterval, edge: 'left' };
          } else {
            return { interval: fakeInterval, edge: 'right' };
          }
        }
      }
    }
    
    return null;
  }, [timelineState, intervals, keyframes, getTrackFromY, getTrackY, getFrameFromX]);

  // Handle mouse events
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!timelineState || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const trackInfo = getTrackFromY(y);
      
      // Header area: drag the time cursor (always allowed)
      if (trackInfo.type === 'header') {
      const frame = getFrameFromX(x);
        dispatchInteraction({ type: "startCursorDrag" });
      onFrameChange?.(frame);
        return;
      }
      
      // If constraints are disabled, don't allow any other interactions
      if (timelineState.constraints_enabled === false) {
        return;
      }
      
      // Prompt area: check for resize, drag, delete, or create
      if (trackInfo.type === 'prompt') {
        const isShiftClick = e.shiftKey;
        const clickedPrompt = findPromptAtPosition(x, y);
        const intersection = findPromptIntersection(x, y);

        // If we're currently editing a prompt and the user clicks empty prompt-space,
        // interpret it as "exit editing" (do not create/append a new prompt).
        if (editingPrompt && !clickedPrompt) {
          // Trigger the existing onBlur handler so edits are committed if needed.
          editingPromptDivRef.current?.blur();
          e.preventDefault();
          return;
        }
        
        // Splittext: Check for intersection interactions first
        if (intersection) {
          // Right-click on intersection: merge prompts
          if (e.button === 2) {
            console.log(`[Timeline] Merging prompts at intersection frame ${intersection.frame}`);
            onPromptMerge?.(intersection.leftPrompt.uuid, intersection.rightPrompt.uuid);
            e.preventDefault();
            return;
          }
          
          // Left-click on intersection: start dragging intersection
          if (e.button === 0) {
            console.log(`[Timeline] Starting intersection drag at frame ${intersection.frame}, left: ${intersection.leftPrompt.uuid}, right: ${intersection.rightPrompt.uuid}`);
            // Snapshot the right-hand chain of prompts so we can shift them all together during drag.
            // This avoids accumulating drift from async state updates and prevents gaps.
            const rightIdx = promptsByStart.findIndex(p => p.uuid === intersection.rightPrompt.uuid);
            const rightChain = rightIdx >= 0
              ? promptsByStart.slice(rightIdx).map(p => ({ uuid: p.uuid, originalStart: p.start_frame, originalEnd: p.end_frame }))
              : [{ uuid: intersection.rightPrompt.uuid, originalStart: intersection.rightPrompt.start_frame, originalEnd: intersection.rightPrompt.end_frame }];
            const rightChainMinStart = rightChain.reduce((acc, p) => Math.min(acc, p.originalStart), Infinity);
            const rightChainMaxEnd = rightChain.reduce((acc, p) => Math.max(acc, p.originalEnd), -Infinity);

            dispatchInteraction({
              type: "startIntersectionDrag",
              payload: {
              leftPromptId: intersection.leftPrompt.uuid,
              rightPromptId: intersection.rightPrompt.uuid,
              originalFrame: intersection.frame,
              leftOriginalStart: intersection.leftPrompt.start_frame,
              rightChain,
              rightChainMinStart,
              rightChainMaxEnd,
            },
            });
            return;
          }
        }
        
        // Splittext: Shift+click on prompt splits it
        if (clickedPrompt && isShiftClick && e.button === 0 && !intersection) {
          const splitFrame = getFrameFromX(x);
          // Only split if click is within the prompt bounds (not on edges)
          if (splitFrame > clickedPrompt.start_frame && splitFrame < clickedPrompt.end_frame) {
            console.log(`[Timeline] Splitting prompt ${clickedPrompt.uuid} at frame ${splitFrame}`);
            onPromptSplit?.(clickedPrompt.uuid, splitFrame);
            return;
          }
        }
        
        // Right-click: delete prompt
        if (clickedPrompt && e.button === 2 && !intersection) {
          // Prevent deleting the last remaining prompt.
          if (timelineState.prompts && timelineState.prompts.length <= 1) {
            console.log(`[Timeline] Cannot delete the last remaining prompt in splittext mode`);
            e.preventDefault();
            return;
          }
          onPromptDelete?.(clickedPrompt.uuid);
          e.preventDefault();
          return;
        }
        
        // Left-click handling
        if (e.button === 0) {
          // Check for prompt resize edge.
          // Splittext: the resize handle is always active (no Ctrl/Cmd), but only for the last prompt's right edge.
          let promptEdge: {prompt: TimelinePrompt, edge: 'left' | 'right'} | null = null;
          const edge = findPromptEdge(x, y);
          if (edge) {
                const lastPrompt = promptsByEndDesc[0];
            if (edge.prompt.uuid === lastPrompt.uuid && edge.edge === 'right') {
              promptEdge = edge;
            }
          }
          
          if (promptEdge) {
            // Start resizing prompt
            const { prompt, edge } = promptEdge;
            dispatchInteraction({
              type: "startPromptResize",
              payload: {
                uuid: prompt.uuid,
                edge: edge,
                originalStartFrame: prompt.start_frame,
                originalEndFrame: prompt.end_frame,
              },
            });
            setResizingPromptCurrentFrames({
              startFrame: prompt.start_frame,
              endFrame: prompt.end_frame,
            });
            return;
          }

          // Splittext: clicking empty space AFTER the last prompt appends a new default prompt.
          // (We keep prompts contiguous by starting at lastPrompt.end_frame.)
          //
          // IMPORTANT: This must run *after* the prompt-edge resize check above; otherwise clicking a few pixels
          // to the right of the last prompt (still inside the resize hitbox / ew-resize cursor) would incorrectly
          // append a new prompt instead of starting a resize.
          if (!isShiftClick && !intersection && !clickedPrompt) {
            const clickFrame = getFrameFromX(x);
            if (prompts.length > 0) {
              const lastPrompt = promptsByEndDesc[0];
              if (clickFrame >= lastPrompt.end_frame) {
                const defaultText = timelineState.default_text ?? "";
                const defaultDuration = timelineState.default_duration ?? 30;
                const minDur = timelineState.min_prompt_duration ?? 1;
                const maxDur = timelineState.max_prompt_duration ?? Infinity;

                const dur = Math.max(minDur, Math.min(maxDur, defaultDuration));
                const startFrame = lastPrompt.end_frame;
                const endFrame = startFrame + dur;
                onPromptAdd?.(startFrame, endFrame, defaultText);
                e.preventDefault();
                return;
              }
            }
          }
        }
      }
      
      // Track area: check for interval and keyframe interaction
      if (trackInfo.type === 'track' && trackInfo.trackIndex !== undefined) {
        const clickedInterval = findIntervalAtPosition(x, y);
        const clickedKeyframe = findKeyframeAtPosition(x, y);
        const isCtrlClick = e.ctrlKey || e.metaKey; // Ctrl on Windows/Linux, Cmd on Mac
        const intervalEdge = isCtrlClick ? findIntervalEdge(x, y) : null;
        
        // Priority 1: Resize mode (Ctrl+click on edge)
        if (intervalEdge && isCtrlClick && e.button === 0) {
          const { interval, edge } = intervalEdge;
          
          // Check if this is a keyframe (needs conversion) or an existing interval
          const isKeyframe = !timelineState.intervals?.some(i => i.uuid === interval.uuid);
          
          dispatchInteraction({
            type: "startIntervalResize",
            payload: {
              uuid: interval.uuid,
              trackId: interval.track_id,
              edge: edge,
              originalStartFrame: interval.start_frame,
              originalEndFrame: interval.end_frame,
              isKeyframe: isKeyframe, // Track if we need to convert keyframe to interval
            },
          });
          setResizingIntervalCurrentFrames({
            startFrame: interval.start_frame,
            endFrame: interval.end_frame,
          });
          return;
        }
        
        // Priority 2: Check keyframes FIRST (they're smaller and drawn on top), then intervals
        if (clickedKeyframe && !isCtrlClick) {
          // Start dragging existing keyframe (only if not Ctrl+click)
          if (e.button === 0) { // Left click
            dispatchInteraction({
              type: "startKeyframeDrag",
              payload: { uuid: clickedKeyframe.uuid, trackId: clickedKeyframe.track_id },
            });
            setDraggedKeyframeFrame(clickedKeyframe.frame); // Show yellow box immediately
          } else if (e.button === 2) { // Right click - delete keyframe
            onKeyframeDelete?.(clickedKeyframe.uuid);
            e.preventDefault();
          }
        } else if (clickedInterval && !isCtrlClick) {
          // Clicked on an interval (only if no keyframe was clicked)
          if (e.button === 0) { // Left click - start dragging interval
            const { interval, clickOffsetFrames } = clickedInterval;
            dispatchInteraction({
              type: "startIntervalMove",
              payload: {
                uuid: interval.uuid,
                trackId: interval.track_id,
                originalStartFrame: interval.start_frame,
                originalEndFrame: interval.end_frame,
                clickOffsetFrames: clickOffsetFrames,
              },
            });
            // Show yellow boxes immediately when clicking on interval
            setDraggedIntervalCurrentFrames({
              startFrame: interval.start_frame,
              endFrame: interval.end_frame
            });
          } else if (e.button === 2) { // Right click - delete interval
            onIntervalDelete?.(clickedInterval.interval.uuid);
            e.preventDefault();
          }
        } else if (e.button === 0) {
          const frame = getFrameFromX(x);
          const track = timelineState.tracks[trackInfo.trackIndex];
          
          if (isCtrlClick) {
            // Ctrl+click: Start creating an interval (centered on the clicked frame)
            console.log(`[Timeline] Starting interval at frame ${frame} on track ${track.name}`);
            dispatchInteraction({
              type: "startIntervalDrag",
              payload: { trackId: track.uuid, startFrame: frame, endFrame: frame },
            });
          } else {
            // Normal click: Add single keyframe
            const tempUuid = onKeyframeAdd?.(track.uuid, frame);
            
            // Immediately start dragging the newly created keyframe
            // Poll for the keyframe to exist in state before starting drag
            if (tempUuid && getLatestState) {
              console.log(`[Timeline] Starting drag for new keyframe ${tempUuid.slice(0, 8)} at frame ${frame}`);
              
              // Poll for keyframe existence using getLatestState (max 10 attempts)
              let attempts = 0;
              const pollInterval = setInterval(() => {
                attempts++;
                const latestState = getLatestState();
                const found = latestState?.keyframes.some(kf => kf.uuid === tempUuid);
                
                console.log(`[Timeline] Poll attempt ${attempts}, found: ${found}, keyframes: ${latestState?.keyframes.length}`);
                
                if (found || attempts >= 10) {
                  clearInterval(pollInterval);
                  if (found) {
                    // Store the initial frame so we can find the real keyframe later
                    dispatchInteraction({
                      type: "startKeyframeDrag",
                      payload: { uuid: tempUuid, trackId: track.uuid, initialFrame: frame },
                    });
                    setDraggedKeyframeFrame(frame);
                  }
                }
              }, 5);
            }
          }
        }
      }
    },
    [timelineState, promptsByStart, promptsByEndDesc, editingPrompt, getFrameFromX, getTrackFromY, findKeyframeAtPosition, findIntervalAtPosition, findIntervalEdge, findPromptEdge, findPromptAtPosition, findPromptIntersection, onFrameChange, onKeyframeAdd, onKeyframeDelete, onIntervalDelete, getLatestState],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!timelineState || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const isCtrlKey = e.ctrlKey || e.metaKey;
      const isShiftKey = e.shiftKey;
      
      // Track if hovering over header (frame numbers area)
      const trackInfo = getTrackFromY(y);
      setIsHoveringHeader(trackInfo.type === 'header');
      setIsHoveringLabelArea(x < TRACK_LABEL_WIDTH);
      
      // If constraints are disabled, only allow cursor dragging
      if (timelineState.constraints_enabled === false) {
        // Clear all hover states
        setHoverEdge(null);
        setHoveredPromptForResize(null);
        setHoveredIntervalForResize(null);
        setHoveredKeyframe(null);
        setHoveredSplitPreview(null);
        setHoveredPromptResize(null);
        
        // Still allow dragging time cursor
        if (isDraggingCursor) {
          const frame = getFrameFromX(x);
          onFrameChange?.(frame);
        }
        return;
      }
      
      // Check for split preview when Shift is held
      if (isShiftKey && trackInfo.type === 'prompt') {
        const clickedPrompt = findPromptAtPosition(x, y);
        if (clickedPrompt) {
          const splitFrame = getFrameFromX(x);
          // Only show preview if within prompt bounds (not on edges)
          if (splitFrame > clickedPrompt.start_frame && splitFrame < clickedPrompt.end_frame) {
            setHoveredSplitPreview({
              promptId: clickedPrompt.uuid,
              frame: splitFrame
            });
          } else {
            setHoveredSplitPreview(null);
          }
        } else {
          setHoveredSplitPreview(null);
        }
      } else {
        setHoveredSplitPreview(null);
      }
      
      // Check for intersection hover
      if (trackInfo.type === 'prompt' && !isShiftKey) {
        const intersection = findPromptIntersection(x, y);
        if (intersection) {
          setHoveredIntersection({
            leftPromptId: intersection.leftPrompt.uuid,
            rightPromptId: intersection.rightPrompt.uuid,
            frame: intersection.frame
          });
        } else {
          setHoveredIntersection(null);
        }
      } else {
        setHoveredIntersection(null);
      }
      
      // Handle intersection dragging
      if (draggingIntersection) {
        // Track pointer position so the RAF loop can continue dragging even if the mouse stops moving.
        autoPanCanvasXRef.current = x;

        const timelineStartX = TRACK_LABEL_WIDTH + 2 * KEYFRAME_RADIUS;
        const leftThreshold = timelineStartX + AUTO_PAN_EDGE_PX;
        const rightThreshold = rect.width - AUTO_PAN_EDGE_PX;
        if (x <= leftThreshold || x >= rightThreshold) {
          startAutoPan("intersection");
        } else {
          stopAutoPan();
        }

        const currentFrame = getFrameFromX(x);
        applyDraggingIntersectionAtFrame(currentFrame);
        return;
      }
      
      // Update hover state for cursor change
      if (!isDraggingCursor && !draggedInterval && !draggedKeyframe && !intervalDrag && !resizingInterval && !resizingPrompt) {
        // Splittext: show the last prompt's RIGHT resize handle without any modifier keys.
        if (trackInfo.type === 'prompt' && !isShiftKey) {
          const edge = findPromptEdge(x, y);
          if (edge) {
          const lastPrompt = promptsByEndDesc[0];
            if (edge.prompt.uuid === lastPrompt.uuid && edge.edge === 'right') {
              setHoverEdge('right');
              setHoveredPromptResize({ promptId: lastPrompt.uuid, frame: lastPrompt.end_frame });
              setHoveredPromptForResize(null);
              setHoveredIntervalForResize(null);
              setHoveredKeyframe(null);
              // Don't let Ctrl-only hover logic override this.
              return;
            }
          }
          setHoveredPromptResize(null);
        } else {
          setHoveredPromptResize(null);
        }

        // Check for interval edges (requires Ctrl/Cmd)
        if (isCtrlKey) {
          const intervalEdge = findIntervalEdge(x, y);
          if (intervalEdge) {
            setHoverEdge(intervalEdge.edge);
            setHoveredIntervalForResize({
              uuid: intervalEdge.interval.uuid,
              edge: intervalEdge.edge,
              startFrame: intervalEdge.interval.start_frame,
              endFrame: intervalEdge.interval.end_frame
            });
            setHoveredPromptForResize(null);
            setHoveredKeyframe(null);
          } else {
          // Check for keyframe hover
          const keyframe = findKeyframeAtPosition(x, y);
          if (keyframe) {
            setHoverEdge(null);
            setHoveredKeyframe({
              uuid: keyframe.uuid,
              frame: keyframe.frame
            });
            setHoveredPromptForResize(null);
            setHoveredIntervalForResize(null);
          } else {
            setHoverEdge(null);
            setHoveredPromptForResize(null);
            setHoveredIntervalForResize(null);
            setHoveredKeyframe(null);
          }
          }
        } else {
          setHoverEdge(null);
          setHoveredPromptForResize(null);
          setHoveredIntervalForResize(null);
          setHoveredKeyframe(null);
        }
      } else {
        setHoverEdge(null);
        setHoveredPromptForResize(null);
        setHoveredIntervalForResize(null);
        setHoveredKeyframe(null);
      }
      
      // Dragging time cursor
      if (isDraggingCursor) {
      const frame = getFrameFromX(x);
      onFrameChange?.(frame);
        return;
      }
      
      // Resizing prompt
      if (resizingPrompt) {
        // Track pointer position so the RAF loop can continue resizing even if the mouse stops moving.
        autoPanCanvasXRef.current = x;

        // If resizing the RIGHT edge and the cursor approaches the right edge of the viewport, auto-pan right.
        if (resizingPrompt.edge === "right") {
          const timelineStartX = TRACK_LABEL_WIDTH + 2 * KEYFRAME_RADIUS;
          const leftThreshold = timelineStartX + AUTO_PAN_EDGE_PX;
          const rightThreshold = rect.width - AUTO_PAN_EDGE_PX;
          if (x <= leftThreshold || x >= rightThreshold) {
            startAutoPan("resizeRightEdge");
          } else {
            stopAutoPan();
          }
        } else {
          stopAutoPan();
        }

        const frame = getFrameFromX(x);
        applyResizingPromptAtFrame(frame);
        return;
      }
      
      // Resizing interval
      if (resizingInterval) {
        const frame = getFrameFromX(x);
        let newStartFrame = resizingInterval.originalStartFrame;
        let newEndFrame = resizingInterval.originalEndFrame;
        
        // Allow dragging past the opposite edge to flip the interval
        if (resizingInterval.edge === 'left') {
          newStartFrame = frame;
        } else {
          newEndFrame = frame;
        }
        
        // Ensure proper ordering for display and storage
        const actualStartFrame = Math.min(newStartFrame, newEndFrame);
        const actualEndFrame = Math.max(newStartFrame, newEndFrame);
        
        // Clamp to timeline bounds
        const timelineStartFrame = timelineState.start_frame;
        const timelineEndFrame = timelineState.end_frame;
        const clampedStartFrame = Math.max(timelineStartFrame, actualStartFrame);
        const clampedEndFrame = Math.min(timelineEndFrame, actualEndFrame);
        
        // Update yellow boxes with clamped ordered frames
        setResizingIntervalCurrentFrames({ startFrame: clampedStartFrame, endFrame: clampedEndFrame });
        
        // Handle keyframe to interval conversion
        if (resizingInterval.isKeyframe && clampedStartFrame !== clampedEndFrame) {
          // First time converting - delete keyframe and create interval
          if (!resizingInterval.convertedIntervalId) {
            console.log('[Timeline] Converting keyframe to interval');
            
            // Delete the keyframe
            if (onKeyframeDelete) {
              onKeyframeDelete(resizingInterval.uuid);
            }
            
            // Create the interval
            if (onIntervalAdd) {
              const newIntervalId = onIntervalAdd(resizingInterval.trackId, clampedStartFrame, clampedEndFrame);
              
              // Update state to track the new interval ID
              dispatchInteraction({
                type: "updateIntervalResize",
                payload: {
                  ...resizingInterval,
                  convertedIntervalId: newIntervalId || resizingInterval.uuid,
                  isKeyframe: false, // No longer a keyframe
                },
              });
            }
          } else {
            // Already converted, just move the interval
            if (onIntervalMove) {
              onIntervalMove(resizingInterval.convertedIntervalId, clampedStartFrame, clampedEndFrame);
            }
          }
        } else if (!resizingInterval.isKeyframe) {
          // Already an interval, just resize it
          if (onIntervalMove) {
            const intervalId = resizingInterval.convertedIntervalId || resizingInterval.uuid;
            onIntervalMove(intervalId, clampedStartFrame, clampedEndFrame);
          }
        }
        
        return;
      }
      
      // Dragging existing interval (moving it)
      if (draggedInterval) {
        const currentFrame = getFrameFromX(x);
        const newStartFrame = currentFrame - draggedInterval.clickOffsetFrames;
        const intervalDuration = draggedInterval.originalEndFrame - draggedInterval.originalStartFrame;
        const newEndFrame = newStartFrame + intervalDuration;
        
        // Clamp interval to timeline bounds
        const timelineStartFrame = timelineState.start_frame;
        const timelineEndFrame = timelineState.end_frame;
        let clampedStartFrame = newStartFrame;
        let clampedEndFrame = newEndFrame;
        
        // If interval would go before timeline start, clamp to start
        if (clampedStartFrame < timelineStartFrame) {
          clampedStartFrame = timelineStartFrame;
          clampedEndFrame = timelineStartFrame + intervalDuration;
        }
        
        // If interval would go after timeline end, clamp to end
        if (clampedEndFrame > timelineEndFrame) {
          clampedEndFrame = timelineEndFrame;
          clampedStartFrame = timelineEndFrame - intervalDuration;
        }
        
        // Update current frames for yellow box display
        setDraggedIntervalCurrentFrames({ startFrame: clampedStartFrame, endFrame: clampedEndFrame });
        
        if (onIntervalMove) {
          onIntervalMove(draggedInterval.uuid, clampedStartFrame, clampedEndFrame);
        }
        return;
      }
      
      // Creating new interval (Ctrl+drag)
      if (intervalDrag) {
        const frame = getFrameFromX(x);
        dispatchInteraction({
          type: "updateIntervalDrag",
          payload: { ...intervalDrag, endFrame: frame },
        });
        return;
      }
      
      // Dragging keyframe
      if (draggedKeyframe) {
        const frame = getFrameFromX(x);
        setDraggedKeyframeFrame(frame);
        
        // If dragging a temp keyframe that got replaced by server, find the real one
        let keyframeId = draggedKeyframe.uuid;
        if (keyframeId.startsWith('temp-') && timelineState && draggedKeyframe.initialFrame !== undefined) {
          // Find the keyframe on the same track at the EXACT initial frame where we created it
          // IMPORTANT: Exclude locked keyframes - we only want to find the newly created unlocked one
          const realKeyframe = timelineState.keyframes.find(kf => 
            kf.track_id === draggedKeyframe.trackId && 
            kf.frame === draggedKeyframe.initialFrame && // Exact match on initial frame
            !kf.locked // Don't pick up locked keyframes
          );
          if (realKeyframe && realKeyframe.uuid !== keyframeId) {
            console.log(`[Timeline] Switching from temp ${keyframeId.slice(0, 8)} to real ${realKeyframe.uuid.slice(0, 8)}`);
            keyframeId = realKeyframe.uuid;
            dispatchInteraction({
              type: "startKeyframeDrag",
              payload: {
                uuid: realKeyframe.uuid,
                trackId: draggedKeyframe.trackId,
                initialFrame: draggedKeyframe.initialFrame,
              },
            });
          }
        }
        
        // Always call the move handler to update position
        if (onKeyframeMove) {
          onKeyframeMove(keyframeId, frame);
        }
      }
    },
    [isDraggingCursor, draggedInterval, intervalDrag, draggedKeyframe, resizingInterval, resizingPrompt, draggingIntersection, promptsByEndDesc, timelineState, getFrameFromX, findIntervalEdge, findPromptEdge, findPromptAtPosition, findPromptIntersection, getTrackFromY, onFrameChange, onIntervalMove, onKeyframeMove, onKeyframeDelete, onIntervalAdd, onPromptResize, onPromptMove],
  );

  const handleMouseUp = useCallback(() => {
    stopAutoPan();
    // Create interval constraint if we were creating one with Ctrl+drag
    if (intervalDrag && onIntervalAdd) {
      const startFrame = Math.min(intervalDrag.startFrame, intervalDrag.endFrame);
      const endFrame = Math.max(intervalDrag.startFrame, intervalDrag.endFrame);
      
      console.log(`[Timeline] Creating interval from ${startFrame} to ${endFrame}`);
      
      // Call the interval add callback
      onIntervalAdd(intervalDrag.trackId, startFrame, endFrame);
    }
    
    // If we were dragging an intersection, we've already been sending resize updates
    // No need to send a final update here since it was happening continuously
    
    // Reset all drag states
    dispatchInteraction({ type: "stopAll" });
    setDraggedKeyframeFrame(null);
    setDraggedIntervalCurrentFrames(null);
    setResizingIntervalCurrentFrames(null);
    setResizingPromptCurrentFrames(null);
    setHoveredPromptResize(null);
    setHoverEdge(null);
    setDraggingIntersectionCurrentFrame(null); // Clear intersection frame indicator
  }, [intervalDrag, onIntervalAdd, stopAutoPan]);

  const handleContextMenu = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // Prevent default context menu
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    if (!timelineState) return;
    
    const { start_frame } = timelineState;
    const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
    const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
    
    // Calculate scroll velocity to detect new gestures
    const now = Date.now();
    const timeDelta = now - lastScrollTimeRef.current;
    const scrollDelta = Math.abs(e.deltaY);
    
    // Calculate velocity (pixels per ms)
    const currentVelocity = timeDelta > 0 ? scrollDelta / timeDelta : 0;
    const lastVelocity = scrollVelocityRef.current;
    
    // Detect new gesture: velocity increases significantly (user actively scrolling)
    // OR first scroll after a long pause
    const isNewGesture = timeDelta > 200 || (lastVelocity > 0 && currentVelocity > lastVelocity * 1.5);
    
    // Update tracking refs
    lastScrollTimeRef.current = now;
    lastScrollDeltaRef.current = scrollDelta;
    scrollVelocityRef.current = currentVelocity;
    
    // Determine scroll mode
    let scrollMode: 'zoom' | 'pan';
    const desiredMode = e.shiftKey ? 'zoom' : 'pan';  // Reversed: scroll = pan, shift+scroll = zoom
    
    if (scrollModeRef.current === null || isNewGesture) {
      // New scroll gesture - lock the mode based on current shift key state
      scrollMode = desiredMode;
      scrollModeRef.current = scrollMode;
    } else {
      // Ongoing scroll (momentum) - keep using locked mode, ignore shift key changes
      scrollMode = scrollModeRef.current;
    }
    
    // Clear any existing timeout and set a new one to detect when scrolling ends
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    scrollTimeoutRef.current = setTimeout(() => {
      scrollModeRef.current = null; // Unlock mode when scrolling stops
      scrollVelocityRef.current = 0; // Reset velocity
    }, 200); // 200ms after last scroll event
    
    if (scrollMode === 'pan') {
      // Pan left/right - allow at any zoom level for infinite scrolling
      e.preventDefault();
      didUserAdjustViewRef.current = true;
      // Use actual scroll speed (deltaY) and scale it to frame units
      const panSpeedScale = 0.1; // Adjust this to control sensitivity
      const panSpeed = e.deltaY * panSpeedScale;
      const newPanOffset = panOffset + panSpeed;
      
      // Allow infinite scrolling to the right, only limit scrolling left
      setPanOffset(Math.max(0, newPanOffset));
    } else {
      // Zoom in/out
      e.preventDefault();
      didUserAdjustViewRef.current = true;
      const zoomSpeed = 0.001;
      const zoomDelta = -e.deltaY * zoomSpeed;
      const maxZoomLevel = Math.max(20, totalFrames);
      const newZoomLevel = Math.max(1, Math.min(maxZoomLevel, zoomLevel * (1 + zoomDelta)));
      
      // Anchor zoom so the current frame (blue box) doesn't move when it's visible.
      // If the current frame isn't visible, fall back to cursor-centered zoom.
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const visibleFrames = totalFrames / zoomLevel;
        const viewStartFrame = start_frame + panOffset;
        const pixelRange = (rect.width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
        const viewEndFrame = viewStartFrame + visibleFrames;

        let anchorFrame: number;
        let anchorRelativePosition: number;
        const currentFrame = timelineState.current_frame;

        if (currentFrame >= viewStartFrame && currentFrame <= viewEndFrame) {
          anchorFrame = currentFrame;
          anchorRelativePosition = (anchorFrame - viewStartFrame) / visibleFrames;
        } else {
          const mouseX = e.clientX - rect.left;
          const adjustedX = mouseX - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS;
          anchorFrame = adjustedX / pixelRange + viewStartFrame;
          anchorRelativePosition = (anchorFrame - viewStartFrame) / visibleFrames;
        }
        
        // Calculate new pan offset to keep the mouse position fixed in frame space
        const newVisibleFrames = totalFrames / newZoomLevel;
        const newViewStartFrame = anchorFrame - anchorRelativePosition * newVisibleFrames;
        const newPanOffset = newViewStartFrame - start_frame;
        
        // Allow infinite scrolling to the right, only limit scrolling left
        setZoomLevel(newZoomLevel);
        setPanOffset(Math.max(0, newPanOffset));
      } else {
        setZoomLevel(newZoomLevel);
      }
    }
  }, [timelineState, zoomLevel, panOffset, canvasSize.width]);

  const handleDoubleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!timelineState || !canvasRef.current || !containerRef.current) return;
    
    // If constraints are disabled, don't allow editing
    if (timelineState.constraints_enabled === false) {
      return;
    }
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if double-clicked on a prompt
    const prompt = findPromptAtPosition(x, y);
    if (prompt) {
      // Calculate prompt position for overlay
      const { start_frame } = timelineState;
      const width = rect.width;
      
      // Calculate visible frame range based on zoom and pan
      const envelopeEndFrame = getZoomEnvelopeEndFrame(timelineState);
      const totalFrames = Math.max(1, envelopeEndFrame - start_frame + 1);
      const visibleFrames = totalFrames / zoomLevel;
      const viewStartFrame = start_frame + panOffset;
      
      const pixelRange = (width - TRACK_LABEL_WIDTH - 2*KEYFRAME_RADIUS) / visibleFrames;
      const timelineStartX = TRACK_LABEL_WIDTH + 2*KEYFRAME_RADIUS;
      
      const startX = timelineStartX + (prompt.start_frame - viewStartFrame) * pixelRange;
      const endX = timelineStartX + (prompt.end_frame - viewStartFrame) * pixelRange;
      const promptWidth = endX - startX;
      // Match the rendered prompt box geometry (see draw loop).
      const promptY = FRAME_LABELS_HEIGHT + 2;
      const promptHeight = TRACK_HEIGHT - 4;
      
      setEditingPrompt({
        uuid: prompt.uuid,
        text: prompt.text,
        x: startX,
        y: promptY,
        width: promptWidth,
        height: promptHeight
      });
    }
  }, [timelineState, findPromptAtPosition, zoomLevel, panOffset]);

  // Set up canvas size
  useEffect(() => {
    if (!containerRef.current || !canvasRef.current) {
      return;
    }
    
    const container = containerRef.current;
    const canvas = canvasRef.current;
    
    const resizeCanvas = () => {
      if (!container || !canvas) return;
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      // Only update if dimensions are non-zero
      if (rect.width === 0 || rect.height === 0) return;

      // Set actual canvas buffer size scaled by DPR for crisp rendering.
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      // Ensure CSS dimensions stay in logical pixels.
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;

      // Trigger redraw by updating state.
      setCanvasSize({ width: rect.width, height: rect.height, dpr });
      setResizeRevision(prev => prev + 1);
    };
    
    // Size immediately
    resizeCanvas();
    
    // Observe the CONTAINER for size changes (not the canvas)
    const resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(container);
    
    // Also listen to window resize events for more reliable updates
    window.addEventListener('resize', resizeCanvas);
    
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', resizeCanvas);
      // Clean up scroll timeout on unmount
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, [timelineState?.enabled]); // Re-run when timeline becomes enabled

  // Focus the editing overlay and place cursor at end for stable visual alignment.
  useEffect(() => {
    if (!editingPrompt) return;
    const el = editingPromptDivRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      try {
        el.focus();
        const sel = window.getSelection?.();
        if (!sel) return;
        const range = document.createRange();
        range.selectNodeContents(el);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
      } catch {
        // Ignore selection errors (e.g., if element unmounted).
      }
    });
  }, [editingPrompt?.uuid]);

  // Draw the static background (prompts, tracks, keyframes) only when they change
  useEffect(() => {
    if (
      !timelineState ||
      !canvasRef.current ||
      canvasSize.width === 0 ||
      canvasSize.height === 0
    ) {
      backgroundCanvasRef.current = null; // Clear background if no timeline
      return;
    }

    // Create or reuse background canvas
    if (!backgroundCanvasRef.current) {
      backgroundCanvasRef.current = document.createElement('canvas');
    }
    
    const bgCanvas = backgroundCanvasRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const bgCtx = bgCanvas.getContext("2d");
    if (!ctx || !bgCtx) return;

    const { prompts, tracks, keyframes } = timelineState;
    const dpr = canvasSize.dpr;

    // Always reset background canvas dimensions to ensure clean redraw on resize
    // Setting width/height automatically clears the canvas
    bgCanvas.width = canvas.width;
    bgCanvas.height = canvas.height;

    // Draw static elements to background canvas
    bgCtx.setTransform(1, 0, 0, 1, 0, 0);
    bgCtx.save();
    bgCtx.scale(dpr, dpr);

    const width = canvasSize.width;
    const height = canvasSize.height;
    
    // Calculate timeline coordinates (reuse across all drawing operations)
    const coords = calculateTimelineCoordinates(width, timelineState, zoomLevel, panOffset);

    // Clear canvas
    bgCtx.fillStyle = theme.backgroundColor;
    bgCtx.fillRect(0, 0, width, height);

    // Track layout
    let currentTrackY = FRAME_LABELS_HEIGHT;
    const trackOrder: Array<{type: 'prompt' | 'track', index: number, data: any}> = [];
    
    // Add prompts track if there are prompts
    if (prompts.length > 0) {
      trackOrder.push({type: 'prompt', index: 0, data: {name: 'Prompts'}});
    }
    
    // Add other tracks
    tracks.forEach((track, index) => {
      trackOrder.push({type: 'track', index, data: track});
    });

    // Draw vertical line at the last frame
    drawEndFrameLine(bgCtx, coords, zoomLevel, width, height, theme);

    // Draw frame ticks FIRST (so they appear behind prompts/keyframes)
    drawFrameLabels(bgCtx, coords, width, height, true, theme);


    // Draw shadow at the bottom of the frame labels header (spanning entire width including labels)
    bgCtx.save();
    const shadowGradient = bgCtx.createLinearGradient(0, FRAME_LABELS_HEIGHT, 0, FRAME_LABELS_HEIGHT + 8);
    shadowGradient.addColorStop(0, theme.headerShadowColor);
    shadowGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    bgCtx.fillStyle = shadowGradient;
    bgCtx.fillRect(0, FRAME_LABELS_HEIGHT, width, 8);
    bgCtx.restore();

    // Draw prompts track (always shown)
    const trackY = currentTrackY;
    
    // Draw track separator (only in timeline area, not in label area)
    bgCtx.strokeStyle = theme.trackSeparatorColor;
    bgCtx.lineWidth = 1;
    bgCtx.beginPath();
    bgCtx.moveTo(TRACK_LABEL_WIDTH, trackY);
    bgCtx.lineTo(width, trackY);
    bgCtx.stroke();

    // Last prompt (by start_frame) uses inclusive end in demo convention; others exclusive.
    const lastPromptStartFrame = prompts.length > 0
      ? Math.max(...prompts.map((p: TimelinePrompt) => p.start_frame))
      : -1;

    // Draw prompts if any exist
    prompts.forEach((prompt: TimelinePrompt) => {
      const color = prompt.color || DEFAULT_PROMPT_COLOR;
      const startX = coords.timelineStartX + (prompt.start_frame - coords.viewStartFrame) * coords.pixelRange;
      const endX = coords.timelineStartX + (prompt.end_frame - coords.viewStartFrame) * coords.pixelRange;
      const promptWidth = endX - startX;

      // Skip only if way off screen (beyond buffer)
      // Allow drawing to the left of timelineStartX for shadows to extend into label area
      if (endX < 0 - DRAW_BUFFER || startX > width + DRAW_BUFFER) {
        return;
      }

      const promptHeight = TRACK_HEIGHT - 4;  // 2px top + 1px bottom margin
      const promptY = trackY + 2;
      const borderRadius = 5;

      // Draw prompt box with rounded corners and gradient
        bgCtx.save();
        bgCtx.beginPath();
        
        // Use roundRect if available, otherwise fallback to regular rect
        if (bgCtx.roundRect) {
          bgCtx.roundRect(startX, promptY, promptWidth, promptHeight, borderRadius);
        } else {
          // Fallback for browsers that don't support roundRect
          bgCtx.rect(startX, promptY, promptWidth, promptHeight);
        }
        bgCtx.clip();

      // Gradient fill
        const gradient = bgCtx.createLinearGradient(0, promptY, 0, promptY + promptHeight);
      gradient.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.9)`);
      gradient.addColorStop(1, `rgba(${color[0] * 0.8}, ${color[1] * 0.8}, ${color[2] * 0.8}, 0.9)`);
        bgCtx.fillStyle = gradient;
        bgCtx.fill();

      // Border
        bgCtx.strokeStyle = theme.promptBorderColor;
        bgCtx.lineWidth = 3.0;
        bgCtx.stroke();

      // Subtle inner highlight
        bgCtx.beginPath();
        if (bgCtx.roundRect) {
          bgCtx.roundRect(startX + 1, promptY + 1, promptWidth - 2, promptHeight - 2, borderRadius - 1);
        } else {
          bgCtx.rect(startX + 1, promptY + 1, promptWidth - 2, promptHeight - 2);
        }
        bgCtx.strokeStyle = `rgba(255, 255, 255, 0.15)`;
        bgCtx.lineWidth = 1;
        bgCtx.stroke();

        bgCtx.restore();

      // Draw seconds label near the left edge of the prompt.
      // UI-only: convert frames -> seconds using timelineState.fps.
      // Match demo convention: last prompt (by start_frame) is [start, end] inclusive,
      // so its duration is (end - start + 1) / fps; other prompts use (end - start) / fps.
      const fps = timelineState.fps ?? 30;
      const isLastPrompt = prompt.start_frame === lastPromptStartFrame;
      const frameCount = isLastPrompt
        ? prompt.end_frame - prompt.start_frame + 1
        : prompt.end_frame - prompt.start_frame;
      const durationSeconds = frameCount / Math.max(1e-9, fps);
      const secondsText = `${Math.max(0, durationSeconds).toFixed(1)} s`;

      // Only draw if there's enough room for the pill + some text space.
      bgCtx.save();
      bgCtx.font = "11px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      const pillPadX = 6;
      const pillW = bgCtx.measureText(secondsText).width + pillPadX * 2;
      const pillH = 16;
      // Clamp to the visible left edge of the *timeline area* (the label overlay covers
      // [0, TRACK_LABEL_WIDTH]). We intentionally do NOT clamp to coords.timelineStartX,
      // because that includes extra padding (2*KEYFRAME_RADIUS) to center keyframe circles.
      // Using coords.timelineStartX here makes the first prompt's seconds pill appear offset
      // relative to the track label separator.
      const visibleStartX = Math.max(startX, TRACK_LABEL_WIDTH);
      const visibleWidth = Math.max(0, endX - visibleStartX);
      const pillX = visibleStartX + 6;
      const pillY = promptY + (promptHeight - pillH) / 2;
      const canDrawPill = visibleWidth > pillW + 24;
      // Used later to keep centered prompt text from overlapping the pill.
      const pillRightX = pillX + pillW + 8;
      if (canDrawPill) {
        // Background pill.
        bgCtx.fillStyle = "rgba(0, 0, 0, 0.28)";
        bgCtx.beginPath();
        if (bgCtx.roundRect) {
          bgCtx.roundRect(pillX, pillY, pillW, pillH, 8);
        } else {
          bgCtx.rect(pillX, pillY, pillW, pillH);
        }
        bgCtx.fill();

        // Text.
        bgCtx.fillStyle = theme.promptTextColor;
        bgCtx.textAlign = "left";
        bgCtx.textBaseline = "middle";
        bgCtx.shadowColor = PROMPT_TEXT_SHADOW;
        bgCtx.shadowBlur = 2;
        bgCtx.shadowOffsetX = 0;
        bgCtx.shadowOffsetY = 1;
        bgCtx.fillText(secondsText, pillX + pillPadX, pillY + pillH / 2);
      }
      bgCtx.restore();

      // Draw prompt text with shadow
        bgCtx.fillStyle = theme.promptTextColor;
        bgCtx.font = "12px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
        bgCtx.textAlign = "center";
        bgCtx.textBaseline = "middle";
      
      // Clip text if needed
      const textY = promptY + promptHeight / 2 + PROMPT_TEXT_Y_OFFSET_PX;
      // When the prompt is partially offscreen, center/truncate based on the visible portion.
      // Left edge is clamped to the timeline area (label overlay covers [0, TRACK_LABEL_WIDTH]).
      const visiblePromptStartX = Math.max(startX, TRACK_LABEL_WIDTH);
      const visiblePromptEndX = Math.min(endX, width);
      const visiblePromptWidth = Math.max(0, visiblePromptEndX - visiblePromptStartX);
      // Compute centered text width, but ensure it never overlaps the seconds pill.
      const textCenterX = visiblePromptStartX + visiblePromptWidth / 2;
      let textLeftLimitX = visiblePromptStartX + 10;
      const textRightLimitX = visiblePromptEndX - 10;
      if (canDrawPill) {
        textLeftLimitX = Math.max(textLeftLimitX, pillRightX);
      }
      const maxTextWidth = Math.max(
        0,
        2 * Math.min(textCenterX - textLeftLimitX, textRightLimitX - textCenterX),
      );
      let displayText = prompt.text || "(Empty)";
      
        if (bgCtx.measureText(displayText).width > maxTextWidth) {
        while (
            bgCtx.measureText(displayText + "...").width > maxTextWidth &&
          displayText.length > 0
        ) {
          displayText = displayText.slice(0, -1);
        }
        displayText += "...";
      }

      if (visiblePromptWidth > 30 && maxTextWidth > 0) {
        // Text shadow for better readability
          bgCtx.shadowColor = PROMPT_TEXT_SHADOW;
          bgCtx.shadowBlur = 3;
          bgCtx.shadowOffsetX = 0;
          bgCtx.shadowOffsetY = 1;
          // Center around the visible portion (do not account for the seconds pill).
          bgCtx.fillText(displayText, textCenterX, textY);
          // Reset shadow completely
          bgCtx.shadowColor = 'transparent';
          bgCtx.shadowBlur = 0;
          bgCtx.shadowOffsetX = 0;
          bgCtx.shadowOffsetY = 0;
        }
      });
    
    currentTrackY += TRACK_HEIGHT;

    // Draw keyframe tracks
    tracks.forEach((track: TimelineTrack) => {
      const trackY = currentTrackY;
      const trackColor = track.color || DEFAULT_KEYFRAME_COLOR;
      const trackHeight = TRACK_HEIGHT * (track.height_scale || 1.0);
      
      // Draw track separator (only in timeline area, not in label area)
      bgCtx.strokeStyle = theme.trackSeparatorColor;
      bgCtx.lineWidth = 1;
      bgCtx.beginPath();
      bgCtx.moveTo(TRACK_LABEL_WIDTH, trackY);
      bgCtx.lineTo(width, trackY);
      bgCtx.stroke();
      
      // Draw intervals for this track (behind keyframes)
      const trackIntervals = intervals.filter((interval: any) => interval.track_id === track.uuid);
      trackIntervals.forEach((interval: any) => {
        const segments = getIntervalSegments(interval, promptRange, trackColor);
        segments.forEach((segment) => {
          const roundLeft = segment.startFrame === interval.start_frame;
          const roundRight = segment.endFrame === interval.end_frame;
          drawInterval(
            bgCtx,
            {
              ...interval,
              start_frame: segment.startFrame,
              end_frame: segment.endFrame,
              opacity: segment.opacity,
            },
            coords,
            trackY,
            trackHeight,
            segment.color,
            theme,
            width,
            roundLeft,
            roundRight,
            true,
            false,
          );
        });
        drawInterval(
          bgCtx,
          interval,
          coords,
          trackY,
          trackHeight,
          trackColor,
          theme,
          width,
          true,
          true,
          false,
          true,
        );
      });
      
      // Draw keyframes for this track (on top of intervals)
      const trackKeyframes = keyframes.filter((kf: TimelineKeyframe) => kf.track_id === track.uuid);
      trackKeyframes.forEach((keyframe: TimelineKeyframe) => {
        const isOutOfRange =
          promptRange !== null &&
          (keyframe.frame < promptRange.startFrame ||
            keyframe.frame > promptRange.endFrame);
        const keyframeColor = isOutOfRange ? OUT_OF_RANGE_CONSTRAINT_COLOR : trackColor;
        const baseOpacity = keyframe.opacity ?? 1.0;
        const keyframeOpacity = isOutOfRange
          ? Math.min(baseOpacity, OUT_OF_RANGE_CONSTRAINT_OPACITY)
          : baseOpacity;
        drawKeyframe(
          bgCtx,
          { ...keyframe, opacity: keyframeOpacity },
          coords,
          trackY,
          trackHeight,
          keyframeColor,
          theme,
          width,
        );
      });
      
      currentTrackY += trackHeight;
    });
    
    // Draw label overlay (background over label area and redraw labels/frame numbers)
    drawLabelOverlay(bgCtx, coords, tracks, prompts.length > 0, width, height, theme);

    // Top separator line  
    bgCtx.strokeStyle = theme.topBorderColor;
    bgCtx.lineWidth = 1;
    bgCtx.beginPath();
    bgCtx.moveTo(0, 0);
    bgCtx.lineTo(width, 0);
    bgCtx.stroke();
    bgCtx.restore();
  }, [
    // Only re-render background when these specific values change
    timelineState?.enabled,
    prompts,
    promptRange,
    tracks,
    keyframes,
    intervals,
    timelineState?.start_frame,
    timelineState?.end_frame,
    promptRange,
    canvasSize.width,
    canvasSize.height,
    canvasSize.dpr,
    resizeRevision, // Force redraw on resize
    zoomLevel, // Redraw on zoom change
    panOffset, // Redraw on pan change
    timelineState?.fps, // Redraw seconds labels when fps changes
    darkMode, // Redraw on theme change
  ]);
  
  // Track when background was last updated (use state so changes trigger re-renders)
  const [backgroundRevision, setBackgroundRevision] = React.useState(0);
  React.useEffect(() => {
    setBackgroundRevision(prev => prev + 1);
  }, [
    prompts,
    tracks,
    keyframes,
    intervals,
  ]);

  // Draw only the cursor when current_frame changes
  useEffect(() => {
    if (
      !timelineState ||
      !canvasRef.current ||
      !backgroundCanvasRef.current ||
      canvasSize.width === 0 ||
      canvasSize.height === 0
    ) {
      return;
    }

    const canvas = canvasRef.current;
    const bgCanvas = backgroundCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { current_frame } = timelineState;
    const dpr = canvasSize.dpr;
    const width = canvasSize.width;
    const height = canvasSize.height;
    
    // Calculate timeline coordinates (reuse across all drawing operations)
    const coords = calculateTimelineCoordinates(width, timelineState, zoomLevel, panOffset);

    // Ensure foreground canvas dimensions match
    if (canvas.width !== bgCanvas.width || canvas.height !== bgCanvas.height) {
      canvas.width = bgCanvas.width;
      canvas.height = bgCanvas.height;
    }

    // Copy background to main canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bgCanvas, 0, 0);

    // Draw cursor on top - but first redraw keyframes/intervals
    ctx.save();
    ctx.scale(dpr, dpr);

    const currentFrameX = coords.timelineStartX + (current_frame - coords.viewStartFrame) * coords.pixelRange;
    
    const { tracks } = timelineState;
    
    // Draw vertical cursor line on top of everything - extend to include frame number row
    ctx.save();
    ctx.strokeStyle = theme.timeCursorColor;
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
        ctx.shadowBlur = 4;
    
      ctx.beginPath();
    ctx.moveTo(currentFrameX, 0);  // Start from top (0) instead of FRAME_LABELS_HEIGHT
    ctx.lineTo(currentFrameX, height);
      ctx.stroke();
    
    ctx.restore();
    
    // Draw label overlay (background over label area and redraw labels/frame numbers)
    drawLabelOverlay(ctx, coords, tracks, prompts.length > 0, width, height, theme);
    
    // Extract coordinates for interactive element drawing
    const { timelineStartX, viewStartFrame, pixelRange } = coords;
    
    // Box dimensions - defined once and used for both blue and yellow boxes
    // Make box fill most of the frame number row (FRAME_LABELS_HEIGHT = 35)
    const BOX_WIDTH = 50;
    const BOX_HEIGHT = 30;
    const BOX_Y = 2;  // Small margin from top
    const BOX_RADIUS = 4;
    
    // Draw cursor handle at top (blue box)
    ctx.save();
    ctx.shadowColor = HIGHLIGHT_SHADOW_COLOR;
    ctx.shadowBlur = HIGHLIGHT_SHADOW_BLUR;
    ctx.shadowOffsetY = HIGHLIGHT_SHADOW_OFFSET_Y;

    ctx.fillStyle = theme.timeCursorColor;
    ctx.strokeStyle = HIGHLIGHT_BORDER_COLOR;
    ctx.lineWidth = HIGHLIGHT_BORDER_WIDTH;

    // Draw handle as a rounded rectangle at the top
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(
        currentFrameX - BOX_WIDTH / 2,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        BOX_RADIUS
      );
    } else {
      ctx.rect(currentFrameX - BOX_WIDTH / 2, BOX_Y, BOX_WIDTH, BOX_HEIGHT);
    }
    ctx.fill();
    ctx.stroke();

    // Draw frame number in blue handle
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 11px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
    ctx.shadowBlur = 2;
    ctx.fillText(String(Math.round(current_frame)), currentFrameX, BOX_Y + BOX_HEIGHT / 2);
        ctx.shadowBlur = 0;

    ctx.restore();
    
    // Draw yellow frame number box when dragging a keyframe (on top of blue box)
    if (draggedKeyframeFrame !== null) {
      const dragFrameX = timelineStartX + (draggedKeyframeFrame - viewStartFrame) * pixelRange;
      drawHighlightFrameBox(
        ctx,
        dragFrameX,
        draggedKeyframeFrame,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        BOX_RADIUS,
      );
    }
    
    // Draw yellow boxes for interval start and end frames when creating interval
    if (intervalDrag) {
      const startFrameX = timelineStartX + (intervalDrag.startFrame - viewStartFrame) * pixelRange;
      const endFrameX = timelineStartX + (intervalDrag.endFrame - viewStartFrame) * pixelRange;

      // Draw both yellow boxes
      [
        { frame: intervalDrag.startFrame, x: startFrameX },
        { frame: intervalDrag.endFrame, x: endFrameX }
      ].forEach(({ frame, x }) => {
        drawHighlightFrameBox(
          ctx,
          x,
          frame,
          BOX_Y,
          BOX_WIDTH,
          BOX_HEIGHT,
          BOX_RADIUS,
        );
      });
    }
    
    // Draw yellow boxes when hovering over prompt edge with Ctrl held
    if (hoveredPromptForResize) {
      const startFrameX = timelineStartX + (hoveredPromptForResize.startFrame - viewStartFrame) * pixelRange;
      const endFrameX = timelineStartX + (hoveredPromptForResize.endFrame - viewStartFrame) * pixelRange;
      
      // Draw both yellow boxes to show which prompt will be resized
      [
        { frame: hoveredPromptForResize.startFrame, x: startFrameX },
        { frame: hoveredPromptForResize.endFrame, x: endFrameX }
      ].forEach(({ frame, x }) => {
        drawHighlightFrameBox(
          ctx,
          x,
          frame,
          BOX_Y,
          BOX_WIDTH,
          BOX_HEIGHT,
          BOX_RADIUS,
        );
      });
    }
    
    // Draw yellow boxes for interval start and end frames when moving existing interval
    if (draggedIntervalCurrentFrames) {
      const startFrameX = timelineStartX + (draggedIntervalCurrentFrames.startFrame - viewStartFrame) * pixelRange;
      const endFrameX = timelineStartX + (draggedIntervalCurrentFrames.endFrame - viewStartFrame) * pixelRange;
      
      // Draw both yellow boxes
      [
        { frame: draggedIntervalCurrentFrames.startFrame, x: startFrameX },
        { frame: draggedIntervalCurrentFrames.endFrame, x: endFrameX }
      ].forEach(({ frame, x }) => {
        drawHighlightFrameBox(
          ctx,
          x,
          frame,
          BOX_Y,
          BOX_WIDTH,
          BOX_HEIGHT,
          BOX_RADIUS,
        );
      });
    }
    
    // Draw yellow boxes when resizing interval
    if (resizingIntervalCurrentFrames) {
      const startFrameX = timelineStartX + (resizingIntervalCurrentFrames.startFrame - viewStartFrame) * pixelRange;
      const endFrameX = timelineStartX + (resizingIntervalCurrentFrames.endFrame - viewStartFrame) * pixelRange;
      
      // Draw both yellow boxes
      [
        { frame: resizingIntervalCurrentFrames.startFrame, x: startFrameX },
        { frame: resizingIntervalCurrentFrames.endFrame, x: endFrameX }
      ].forEach(({ frame, x }) => {
        drawHighlightFrameBox(
          ctx,
          x,
          frame,
          BOX_Y,
          BOX_WIDTH,
          BOX_HEIGHT,
          BOX_RADIUS,
        );
      });
    }
    
    // Draw yellow boxes when resizing prompt
    if (resizingPromptCurrentFrames) {
      const startFrameX = timelineStartX + (resizingPromptCurrentFrames.startFrame - viewStartFrame) * pixelRange;
      const endFrameX = timelineStartX + (resizingPromptCurrentFrames.endFrame - viewStartFrame) * pixelRange;
      
      // Draw both yellow boxes
      [
        { frame: resizingPromptCurrentFrames.startFrame, x: startFrameX },
        { frame: resizingPromptCurrentFrames.endFrame, x: endFrameX }
      ].forEach(({ frame, x }) => {
        drawHighlightFrameBox(
          ctx,
          x,
          frame,
          BOX_Y,
          BOX_WIDTH,
          BOX_HEIGHT,
          BOX_RADIUS,
        );
      });
    }
    
    // Draw yellow boxes when hovering over interval edge with Ctrl held
    if (hoveredIntervalForResize) {
      const startFrameX = timelineStartX + (hoveredIntervalForResize.startFrame - viewStartFrame) * pixelRange;
      const endFrameX = timelineStartX + (hoveredIntervalForResize.endFrame - viewStartFrame) * pixelRange;
      
      // Draw both yellow boxes to show which interval will be resized
      [
        { frame: hoveredIntervalForResize.startFrame, x: startFrameX },
        { frame: hoveredIntervalForResize.endFrame, x: endFrameX }
      ].forEach(({ frame, x }) => {
        drawHighlightFrameBox(
          ctx,
          x,
          frame,
          BOX_Y,
          BOX_WIDTH,
          BOX_HEIGHT,
          BOX_RADIUS,
        );
      });
    }
    
    // Draw yellow box when hovering over keyframe with Ctrl held
    if (hoveredKeyframe) {
      const frameX = timelineStartX + (hoveredKeyframe.frame - viewStartFrame) * pixelRange;
      drawHighlightFrameBox(
        ctx,
        frameX,
        hoveredKeyframe.frame,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        BOX_RADIUS,
      );
    }
    
    // Draw interval being created (Ctrl+drag)
    if (intervalDrag) {
      const actualStartFrame = Math.min(intervalDrag.startFrame, intervalDrag.endFrame);
      const actualEndFrame = Math.max(intervalDrag.startFrame, intervalDrag.endFrame);
      
      // Find the track to get its Y position and color
      const trackIndex = timelineState.tracks.findIndex(t => t.uuid === intervalDrag.trackId);
      if (trackIndex !== -1) {
        const track = timelineState.tracks[trackIndex];
        const trackColor = track.color || DEFAULT_KEYFRAME_COLOR;
        const trackHeight = TRACK_HEIGHT * (track.height_scale || 1.0);
        
        // Calculate trackY by accumulating heights of all previous tracks
        let trackY = FRAME_LABELS_HEIGHT;
        // Prompt track is always shown
        trackY += TRACK_HEIGHT;
        for (let i = 0; i < trackIndex; i++) {
          trackY += TRACK_HEIGHT * (timelineState.tracks[i].height_scale || 1.0);
        }
        
        const intervalPreview = {
          start_frame: actualStartFrame,
          end_frame: actualEndFrame,
        };
        const segments = getIntervalSegments(intervalPreview, promptRange, trackColor);
        segments.forEach((segment) => {
          const roundLeft = segment.startFrame === intervalPreview.start_frame;
          const roundRight = segment.endFrame === intervalPreview.end_frame;
          drawInterval(
            ctx,
            {
              ...intervalPreview,
              start_frame: segment.startFrame,
              end_frame: segment.endFrame,
              opacity: segment.opacity,
            },
            coords,
            trackY,
            trackHeight,
            segment.color,
            theme,
            width,
            roundLeft,
            roundRight,
            true,
            false,
          );
        });
        drawInterval(
          ctx,
          intervalPreview,
          coords,
          trackY,
          trackHeight,
          trackColor,
          theme,
          width,
          true,
          true,
          false,
          true,
        );
      }
    }
    
    // Draw split preview (Shift+hover) - yellow box in time row only
    if (hoveredSplitPreview) {
      const splitX = timelineStartX + (hoveredSplitPreview.frame - viewStartFrame) * pixelRange;

      // Draw vertical yellow line in the prompt row to show where the split will occur.
      {
        const promptY = FRAME_LABELS_HEIGHT + 2;
        const promptHeight = TRACK_HEIGHT - 4;
        ctx.save();
        ctx.strokeStyle = HIGHLIGHT_COLOR;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(splitX, promptY);
        ctx.lineTo(splitX, promptY + promptHeight);
        ctx.stroke();
        ctx.restore();
      }
      
      // Draw frame number indicator box in the time row (same style as resize/intersection)
      const BOX_WIDTH = 50;
      const BOX_HEIGHT = 30;
      const BOX_Y = 2;
      
      drawHighlightFrameBox(
        ctx,
        splitX,
        hoveredSplitPreview.frame,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        4,
      );
    }

    // Remove intersection-drag preview in the prompt band (we only show yellow box in time row now)
    // (This section intentionally left empty - visual moved to time row)
    
    // Draw intersection hover indicator removed (no blue line+circle)
    // (Cursor will still change to indicate draggable)

    // Yellow box indicator for resizing last prompt (in time row)
    // IMPORTANT: when actively resizing, prefer the live resizing frames (not the hovered frame),
    // otherwise the box will "stick" at the previous end frame.
    const resizeHandleFrame =
      resizingPromptCurrentFrames
        ? (resizingPrompt?.edge === 'left'
            ? resizingPromptCurrentFrames.startFrame
            : resizingPromptCurrentFrames.endFrame)
        : (hoveredPromptResize?.frame ?? null);

    if (resizeHandleFrame !== null && resizeHandleFrame !== undefined) {
      const handleX = timelineStartX + (resizeHandleFrame - viewStartFrame) * pixelRange;
      const BOX_WIDTH = 50;
      const BOX_HEIGHT = 30;
      const BOX_Y = 2;
      drawHighlightFrameBox(
        ctx,
        handleX,
        resizeHandleFrame,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        4,
      );
    }
    
    // Draw yellow frame indicators during intersection dragging (in frame number line)
    // We show:
    // - the moving intersection frame (acts like resizing the left prompt's right edge)
    // - the left prompt's start frame (to match the "two yellow boxes" visual during resizing)
    if (draggingIntersection && draggingIntersectionCurrentFrame !== null) {
      // Box dimensions - match all other yellow boxes
      const BOX_WIDTH = 50;
      const BOX_HEIGHT = 30;
      const BOX_Y = 2;  // Small margin from top
      
      const intersectionX = timelineStartX + (draggingIntersectionCurrentFrame - viewStartFrame) * pixelRange;
      const leftStartX = timelineStartX + (draggingIntersection.leftOriginalStart - viewStartFrame) * pixelRange;

      // Match "resizing" behavior: show start + end (intersection) boxes.
      drawHighlightFrameBox(
        ctx,
        leftStartX,
        draggingIntersection.leftOriginalStart,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        4,
      );
      drawHighlightFrameBox(
        ctx,
        intersectionX,
        draggingIntersectionCurrentFrame,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        4,
      );
    }
    
    // Draw yellow frame indicator when hovering over intersection (in frame number line)
    if (hoveredIntersection && !draggingIntersection) {
      // Box dimensions - match all other yellow boxes
      const BOX_WIDTH = 50;
      const BOX_HEIGHT = 30;
      const BOX_Y = 2;  // Small margin from top
      
      const intersectionX = timelineStartX + (hoveredIntersection.frame - viewStartFrame) * pixelRange;
      
      drawHighlightFrameBox(
        ctx,
        intersectionX,
        hoveredIntersection.frame,
        BOX_Y,
        BOX_WIDTH,
        BOX_HEIGHT,
        4,
      );
    }
  }, [
    // Only re-render cursor when frame position actually changes
    timelineState?.enabled, // Trigger when timeline first appears
    timelineState?.current_frame,
    timelineState?.start_frame,
    timelineState?.end_frame,
    canvasSize.width,
    canvasSize.height,
    canvasSize.dpr,
    resizeRevision, // Force redraw on resize
    backgroundRevision, // Force redraw when background changes (keyframes added/moved/deleted)
    draggedKeyframeFrame, // Force redraw when dragging keyframe to show yellow box
    intervalDrag, // Force redraw when creating interval
    draggedIntervalCurrentFrames, // Force redraw when moving interval to show yellow boxes
    resizingIntervalCurrentFrames, // Force redraw when resizing interval to show yellow boxes
    resizingPromptCurrentFrames, // Force redraw when resizing prompt to show yellow boxes
    hoveredPromptForResize, // Force redraw when hovering over prompt edge with Ctrl to show yellow boxes
    hoveredIntervalForResize, // Force redraw when hovering over interval edge with Ctrl to show yellow boxes
    hoveredKeyframe, // Force redraw when hovering over keyframe with Ctrl to show yellow box
    hoveredSplitPreview, // Force redraw when hovering with Shift to show split preview
    hoveredIntersection, // Force redraw when hovering over intersection
    draggingIntersection, // Force redraw when dragging intersection
    draggingIntersectionCurrentFrame, // Force redraw with yellow box during intersection drag
    zoomLevel, // Force redraw on zoom change
    panOffset, // Force redraw on pan change
    darkMode, // Force redraw on theme change
  ]);

  if (!timelineState || !timelineState.enabled) {
    return null;
  }

  // Calculate height based on number of tracks and their individual height scales
  let tracksHeight = 0;
  // Prompts row is always shown
  tracksHeight += TRACK_HEIGHT;
  for (const track of timelineState.tracks) {
    tracksHeight += TRACK_HEIGHT * (track.height_scale || 1.0);
  }
  const containerHeight = FRAME_LABELS_HEIGHT + tracksHeight;

  return (
    <div
      ref={containerRef}
      style={{
        position: "fixed",
        bottom: 0,
        left: 0,
        right: 0,
        height: `${containerHeight}px`,
        zIndex: 5,
        margin: 0,
        padding: 0,
        borderTop: `2px solid ${darkMode ? "rgba(0, 0, 0, 0.2)" : "rgba(0, 0, 0, 0.1)"}`,
        boxShadow: darkMode ? "0 -2px 8px 0 rgba(0,0,0,0.15)" : "0 -2px 8px 0 rgba(0,0,0,0.08)",
        backgroundColor: theme.backgroundColor,
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={handleContextMenu}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
        style={{
          width: "100%",
          height: "100%",
          cursor: timelineState.constraints_enabled 
            ? (
              resizingPrompt || resizingPromptCurrentFrames
                ? "ew-resize"
                : isHoveringLabelArea
                  ? "default"
                : (hoveredIntersection || draggingIntersection)
                  ? "ew-resize"
                  : hoverEdge
                    ? "ew-resize"
                    : (isDraggingCursor || draggedKeyframe || draggedInterval)
                      ? "grabbing"
                      : "pointer"
            )
            : (isHoveringHeader || isDraggingCursor ? (isDraggingCursor ? "grabbing" : "pointer") : "default"),
          display: "block",
        }}
      />
      
      {/* Gray overlay when constraints are disabled */}
      {timelineState.constraints_enabled === false && (
        <div
          style={{
            position: "absolute",
            top: `${FRAME_LABELS_HEIGHT}px`,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(128, 128, 128, 0.3)",
            pointerEvents: "none",
            zIndex: 6,
          }}
        />
      )}
      
      {/* Prompt text editing overlay */}
      {editingPrompt && (() => {
        // Find the prompt to get its color
        const prompt = timelineState?.prompts.find(p => p.uuid === editingPrompt.uuid);
        const color = prompt?.color || DEFAULT_PROMPT_COLOR;
        const borderColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        
        return (
          <>
            {/* Click-catcher: clicking anywhere outside the editable prompt exits editing mode and prevents canvas interactions */}
            <div
              style={{
                position: "absolute",
                inset: 0,
                zIndex: 9,
                background: "transparent",
              }}
              onMouseDown={(e) => {
                // If click is outside the editor, blur it (triggers commit via onBlur).
                const editor = editingPromptDivRef.current;
                if (editor && !editor.contains(e.target as Node)) {
                  editor.blur();
                }
                // Always prevent the canvas from seeing the click while editing.
                e.preventDefault();
                e.stopPropagation();
              }}
              onDoubleClick={(e) => {
                // Prevent accidental re-entering edit or creating interactions behind the overlay.
                e.preventDefault();
                e.stopPropagation();
              }}
            />

            <div
              // Remount per prompt to avoid React re-render clobbering caret position while editing.
              key={editingPrompt.uuid}
              style={{
                position: "absolute",
                left: `${editingPrompt.x}px`,
                top: `${editingPrompt.y}px`,
                width: `${editingPrompt.width}px`,
                height: `${editingPrompt.height}px`,
                fontSize: "12px",
                fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                // True vertical centering (avoids UA input baseline quirks).
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxSizing: "border-box",
                border: "none",
                boxShadow: `inset 0 0 0 2px ${borderColor}`,
                borderRadius: "5px",
                outline: "none",
                backgroundColor: darkMode ? "rgba(44, 46, 51, 0.95)" : "rgba(44, 46, 51, 0.95)",
                color: theme.promptTextColor,
                padding: 0,
                zIndex: 10,
              }}
            >
              <div
                ref={editingPromptDivRef}
                contentEditable
                suppressContentEditableWarning
                onBlur={(e) => {
                  const nextText = (e.currentTarget.textContent ?? "");
                  if (onPromptUpdate && nextText !== editingPrompt.text) {
                    onPromptUpdate(editingPrompt.uuid, nextText);
                    // Small delay to let optimistic update propagate before hiding input
                    setTimeout(() => setEditingPrompt(null), 0);
                  } else {
                    setEditingPrompt(null);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    (e.currentTarget as HTMLDivElement).blur();
                  } else if (e.key === "Escape") {
                    e.preventDefault();
                    setEditingPrompt(null);
                  }
                }}
                style={{
                  flex: 1,
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  padding: "0 8px",
                  whiteSpace: "nowrap",
                  overflowX: "auto",
                  overflowY: "hidden",
                  outline: "none",
                }}
              >
                {editingPrompt.text}
              </div>
            </div>
          </>
        );
      })()}
    </div>
  );
}
