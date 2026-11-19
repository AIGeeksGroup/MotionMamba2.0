// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface TimelineTheme {
  timeCursorColor: string;
  backgroundColor: string;
  frameLabelColor: string;
  frameTickColor: string;
  promptBorderColor: string;
  promptTextColor: string;
  trackLabelColor: string;
  trackSeparatorColor: string;
  labelOverlayColor: string;
  keyframeBorderColor: string;
  intervalBorderColor: string;
  headerShadowColor: string;
  topBorderColor: string;
}

export const DARK_THEME: TimelineTheme = {
  timeCursorColor: "#52A5F4",
  backgroundColor: "#2C2E33",
  frameLabelColor: "#A6A7AB",
  frameTickColor: "#41434A",
  promptBorderColor: "#1A1B1E",
  promptTextColor: "#FFFFFF",
  trackLabelColor: "#A6A7AB",
  trackSeparatorColor: "#41434A",
  labelOverlayColor: "#2C2E33",
  keyframeBorderColor: "#FFFFFF",
  intervalBorderColor: "#FFFFFF",
  headerShadowColor: "rgba(0, 0, 0, 0.2)",
  topBorderColor: "rgba(0, 0, 0, 0.3)",
};

export const LIGHT_THEME: TimelineTheme = {
  timeCursorColor: "#2563EB",
  backgroundColor: "#F8F9FA",
  frameLabelColor: "#495057",
  frameTickColor: "#CED4DA",
  promptBorderColor: "#495057",
  promptTextColor: "#FFFFFF",
  trackLabelColor: "#495057",
  trackSeparatorColor: "#CED4DA",
  labelOverlayColor: "#F8F9FA",
  keyframeBorderColor: "#495057",
  intervalBorderColor: "#495057",
  headerShadowColor: "rgba(0, 0, 0, 0.08)",
  topBorderColor: "rgba(0, 0, 0, 0.05)",
};

export const PROMPT_COLORS: [number, number, number][] = [
  [82, 133, 166],  // Default (teal-blue)
  [239, 68, 68],   // Red
  [249, 115, 22],  // Orange
  [234, 179, 8],   // Yellow
  [34, 197, 94],   // Green
  [59, 130, 246],  // Blue
  [168, 85, 247],  // Purple
  [236, 72, 153],  // Pink
];

export const DEFAULT_PROMPT_COLOR: [number, number, number] = PROMPT_COLORS[0]!;
export const PROMPT_TEXT_SHADOW = "rgba(0, 0, 0, 0.5)";

export const HIGHLIGHT_COLOR = "#F4D03F";
export const HIGHLIGHT_BORDER_COLOR = "rgba(255, 255, 255, 0.9)";
export const HIGHLIGHT_SHADOW_COLOR = "rgba(0, 0, 0, 0.3)";
export const HIGHLIGHT_TEXT_COLOR = "#000000";
export const HIGHLIGHT_FONT =
  "bold 10px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
export const HIGHLIGHT_TEXT_SHADOW_COLOR = "rgba(255, 255, 255, 0.6)";
export const HIGHLIGHT_SHADOW_BLUR = 5;
export const HIGHLIGHT_SHADOW_OFFSET_Y = 2;
export const HIGHLIGHT_BORDER_WIDTH = 1.5;
export const HIGHLIGHT_TEXT_SHADOW_BLUR = 2;
