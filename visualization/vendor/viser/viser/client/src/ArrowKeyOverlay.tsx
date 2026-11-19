// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { TimelineMessage } from "./WebsocketMessages";
import { ViewerContext } from "./ViewerContext";
import { FRAME_LABELS_HEIGHT, TRACK_HEIGHT } from "./TimelineConstants";

const ARROW_KEYCAP_SIZE_PX = 46;
const ARROW_KEYCAP_GAP_PX = 7;
const ARROW_OVERLAY_PADDING_PX = 10;

function getTimelineHeight(timelineState: TimelineMessage | null): number {
  if (!timelineState?.enabled) {
    return 0;
  }
  let tracksHeight = TRACK_HEIGHT; // Prompts row is always shown
  for (const track of timelineState.tracks) {
    tracksHeight += TRACK_HEIGHT * (track.height_scale || 1.0);
  }
  return FRAME_LABELS_HEIGHT + tracksHeight;
}

export function ArrowKeyOverlay() {
  const viewer = React.useContext(ViewerContext)!;
  const darkMode = viewer?.useGui((state) => state.theme.dark_mode) ?? true;
  const arrowOverlay = viewer?.useGui((state) => state.arrowKeyOverlay) ?? null;
  const timelineState = viewer?.useGui((state) => state.timeline) ?? null;

  if (arrowOverlay?.enabled !== true) {
    return null;
  }

  const overlayPosition = arrowOverlay.position ?? "bottom_left";
  const bottomOffsetPx =
    getTimelineHeight(timelineState) + ARROW_OVERLAY_PADDING_PX;

  const overlayStyle: React.CSSProperties = (() => {
    const base: React.CSSProperties = {
      position: "fixed",
      zIndex: 7,
      pointerEvents: "none",
    };
    switch (overlayPosition) {
      case "top_center":
        return {
          ...base,
          top: ARROW_OVERLAY_PADDING_PX,
          left: "50%",
          transform: "translateX(-50%)",
        };
      case "top_right":
        return {
          ...base,
          top: ARROW_OVERLAY_PADDING_PX,
          right: "8px",
        };
      case "bottom_center":
        return {
          ...base,
          bottom: `${bottomOffsetPx}px`,
          left: "50%",
          transform: "translateX(-50%)",
        };
      case "bottom_right":
        return {
          ...base,
          bottom: `${bottomOffsetPx}px`,
          right: "8px",
        };
      case "bottom_left":
      default:
        return {
          ...base,
          bottom: `${bottomOffsetPx}px`,
          left: "8px",
        };
    }
  })();

  const highlighted = new Set(arrowOverlay.highlighted);
  const baseOpacity = arrowOverlay.base_opacity ?? 1.0;
  const highlightOpacity = arrowOverlay.highlight_opacity ?? 1.0;

  // Base vs highlight styling.
  // We keep the keycaps fully opaque (solid background) and use color to indicate "pressed".
  const baseBorder = darkMode ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.25)";
  const baseBg = darkMode ? "rgb(24,24,27)" : "rgb(255,255,255)";
  const baseFg = darkMode
    ? `rgba(255,255,255,${baseOpacity})`
    : `rgba(0,0,0,${baseOpacity})`;

  // A light yellow highlight that reads well on both themes.
  const highlightBg = darkMode ? "rgb(250,204,21)" : "rgb(254,243,199)"; // amber-400 / amber-100
  const highlightBorder = darkMode ? "rgb(245,158,11)" : "rgb(217,119,6)"; // amber-500 / amber-600
  const highlightFg = `rgba(0,0,0,${highlightOpacity})`;

  const Keycap = (props: {
    keyName: "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";
    label: string;
    gridColumn: string;
    gridRow: string;
  }) => {
    const isHighlighted = highlighted.has(props.keyName);
    const border = isHighlighted ? highlightBorder : baseBorder;
    const bg = isHighlighted ? highlightBg : baseBg;
    const fg = isHighlighted ? highlightFg : baseFg;
    return (
      <div
        style={{
          gridColumn: props.gridColumn,
          gridRow: props.gridRow,
          width: ARROW_KEYCAP_SIZE_PX,
          height: ARROW_KEYCAP_SIZE_PX,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderRadius: 6,
          border: `1px solid ${border}`,
          background: bg,
          color: fg,
          fontSize: 26,
          lineHeight: "26px",
          fontFamily:
            "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          boxShadow: darkMode
            ? (isHighlighted
              ? "0 6px 16px rgba(0,0,0,0.35)"
              : "0 2px 8px rgba(0,0,0,0.25)")
            : (isHighlighted
              ? "0 6px 16px rgba(0,0,0,0.16)"
              : "0 2px 8px rgba(0,0,0,0.12)"),
        }}
      >
        {props.label}
      </div>
    );
  };

  return (
    <div style={overlayStyle}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `${ARROW_KEYCAP_SIZE_PX}px ${ARROW_KEYCAP_SIZE_PX}px ${ARROW_KEYCAP_SIZE_PX}px`,
          gridTemplateRows: `${ARROW_KEYCAP_SIZE_PX}px ${ARROW_KEYCAP_SIZE_PX}px`,
          gap: ARROW_KEYCAP_GAP_PX,
        }}
      >
        <Keycap keyName="ArrowUp" label="↑" gridColumn="2" gridRow="1" />
        <Keycap keyName="ArrowLeft" label="←" gridColumn="1" gridRow="2" />
        <Keycap keyName="ArrowDown" label="↓" gridColumn="2" gridRow="2" />
        <Keycap keyName="ArrowRight" label="→" gridColumn="3" gridRow="2" />
      </div>
    </div>
  );
}
