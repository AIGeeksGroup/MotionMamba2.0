// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TimelineMessage } from "../WebsocketMessages";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function normalizeTimelineMessage(message: TimelineMessage): TimelineMessage {
  const minPromptDuration = Math.max(1, message.min_prompt_duration ?? 1);
  const maxPromptDuration =
    message.max_prompt_duration == null
      ? null
      : Math.max(minPromptDuration, message.max_prompt_duration);

  const prompts = (message.prompts ?? []).map((prompt) => {
    const start = Math.min(prompt.start_frame, prompt.end_frame);
    let end = Math.max(prompt.start_frame, prompt.end_frame);
    const minEnd = start + minPromptDuration;
    end = Math.max(end, minEnd);
    if (maxPromptDuration != null) {
      end = Math.min(end, start + maxPromptDuration);
    }
    return { ...prompt, start_frame: start, end_frame: end };
  });

  prompts.sort((a, b) => a.start_frame - b.start_frame);

  const defaultNumFramesZoom = Math.max(1, message.default_num_frames_zoom ?? 1);
  const maxFramesZoom = Math.max(defaultNumFramesZoom, message.max_frames_zoom ?? defaultNumFramesZoom);
  const startFrame = message.start_frame ?? 0;
  const endFrame = Math.max(message.end_frame ?? startFrame, startFrame);
  const fps = message.fps > 0 ? message.fps : 30.0;
  const currentFrame = clamp(message.current_frame ?? startFrame, startFrame, endFrame);

  return {
    ...message,
    prompts,
    min_prompt_duration: minPromptDuration,
    max_prompt_duration: maxPromptDuration,
    default_num_frames_zoom: defaultNumFramesZoom,
    max_frames_zoom: maxFramesZoom,
    start_frame: startFrame,
    end_frame: endFrame,
    fps,
    current_frame: currentFrame,
  };
}
