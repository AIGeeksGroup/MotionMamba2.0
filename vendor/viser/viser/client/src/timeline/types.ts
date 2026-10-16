// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface TimelinePrompt {
  uuid: string;
  text: string;
  start_frame: number;
  end_frame: number;
  color: [number, number, number] | null;
}

export interface TimelineTrack {
  uuid: string;
  name: string;
  track_type: string;
  color: [number, number, number] | null;
  height_scale?: number; // Height scale multiplier (default 1.0)
}

export interface TimelineKeyframe {
  uuid: string;
  track_id: string;
  frame: number;
  value: number | null;
  opacity?: number; // Opacity of the keyframe (0.0 to 1.0)
  locked?: boolean; // Whether keyframe is locked from UI modifications
}

export type CanvasSizeState = { width: number; height: number; dpr: number };

export interface TimelineCoordinates {
  pixelRange: number;
  timelineStartX: number;
  viewStartFrame: number;
  viewEndFrame: number;
  visibleFrames: number;
  minFramePixels: number;
  frameDelta: number;
  framesBeforeTimeline: number;
  firstVisibleFrame: number;
  endFrame: number; // End frame for the zoom envelope (used for coordinate scaling)
}
