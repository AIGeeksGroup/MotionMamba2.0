// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FRAME_LABELS_HEIGHT, TRACK_HEIGHT } from "../TimelineConstants";
import { DRAW_BUFFER, KEYFRAME_RADIUS, TRACK_LABEL_WIDTH } from "./constants";
import {
  HIGHLIGHT_BORDER_COLOR,
  HIGHLIGHT_BORDER_WIDTH,
  HIGHLIGHT_COLOR,
  HIGHLIGHT_FONT,
  HIGHLIGHT_SHADOW_BLUR,
  HIGHLIGHT_SHADOW_COLOR,
  HIGHLIGHT_SHADOW_OFFSET_Y,
  HIGHLIGHT_TEXT_COLOR,
  HIGHLIGHT_TEXT_SHADOW_BLUR,
  HIGHLIGHT_TEXT_SHADOW_COLOR,
  TimelineTheme,
} from "./styles";
import { TimelineCoordinates, TimelineKeyframe, TimelineTrack } from "./types";

export function drawHighlightFrameBox(
  ctx: CanvasRenderingContext2D,
  x: number,
  frame: number,
  boxY: number,
  boxWidth: number,
  boxHeight: number,
  boxRadius: number,
): void {
  ctx.save();
  ctx.shadowColor = HIGHLIGHT_SHADOW_COLOR;
  ctx.shadowBlur = HIGHLIGHT_SHADOW_BLUR;
  ctx.shadowOffsetY = HIGHLIGHT_SHADOW_OFFSET_Y;

  ctx.fillStyle = HIGHLIGHT_COLOR;
  ctx.strokeStyle = HIGHLIGHT_BORDER_COLOR;
  ctx.lineWidth = HIGHLIGHT_BORDER_WIDTH;

  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(
      x - boxWidth / 2,
      boxY,
      boxWidth,
      boxHeight,
      boxRadius,
    );
  } else {
    ctx.rect(x - boxWidth / 2, boxY, boxWidth, boxHeight);
  }
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = HIGHLIGHT_TEXT_COLOR;
  ctx.font = HIGHLIGHT_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.shadowColor = HIGHLIGHT_TEXT_SHADOW_COLOR;
  ctx.shadowBlur = HIGHLIGHT_TEXT_SHADOW_BLUR;
  ctx.fillText(String(Math.round(frame)), x, boxY + boxHeight / 2);
  ctx.shadowBlur = 0;
  ctx.restore();
}

export function drawInterval(
  ctx: CanvasRenderingContext2D,
  interval: any,
  coords: TimelineCoordinates,
  trackY: number,
  trackHeight: number,
  trackColor: [number, number, number],
  theme: TimelineTheme,
  width?: number,
  roundLeft: boolean = true,
  roundRight: boolean = true,
  drawFill: boolean = true,
  drawBorder: boolean = true,
): void {
  const intervalFrameWidth = interval.end_frame - interval.start_frame + 1;
  const frameStartX =
    coords.timelineStartX +
    (interval.start_frame - coords.viewStartFrame) * coords.pixelRange;

  const isSingleFrame = intervalFrameWidth === 1;
  const intervalWidth = isSingleFrame
    ? KEYFRAME_RADIUS * 2
    : (interval.end_frame - interval.start_frame) * coords.pixelRange +
      2 * KEYFRAME_RADIUS;

  const rectStartX = frameStartX - KEYFRAME_RADIUS;
  const endX = rectStartX + intervalWidth;

  if (
    width !== undefined &&
    (endX < 0 - DRAW_BUFFER || rectStartX > width + DRAW_BUFFER)
  ) {
    return;
  }

  const intervalHeight = KEYFRAME_RADIUS * 2;
  const intervalY = trackY + (trackHeight - intervalHeight) / 2;
  const opacity = interval.opacity !== undefined ? interval.opacity : 1.0;

  ctx.save();
  ctx.fillStyle = `rgba(${trackColor[0]}, ${trackColor[1]}, ${trackColor[2]}, ${opacity})`;
  ctx.strokeStyle = theme.intervalBorderColor;
  ctx.lineWidth = 1.5;

  const centerY = trackY + trackHeight / 2;
  const centerX = frameStartX;
  const shouldDrawCircle = isSingleFrame && roundLeft && roundRight;
  ctx.beginPath();
  if (shouldDrawCircle) {
    ctx.arc(centerX, centerY, KEYFRAME_RADIUS, 0, Math.PI * 2);
  } else {
    const leftRadius = roundLeft ? KEYFRAME_RADIUS : 0;
    const rightRadius = roundRight ? KEYFRAME_RADIUS : 0;

    if (ctx.roundRect && leftRadius === rightRadius) {
      ctx.roundRect(
        rectStartX,
        intervalY,
        intervalWidth,
        intervalHeight,
        leftRadius,
      );
    } else {
      const x = rectStartX;
      const y = intervalY;
      const w = intervalWidth;
      const h = intervalHeight;
      const rLeft = Math.max(0, Math.min(leftRadius, h / 2, w / 2));
      const rRight = Math.max(0, Math.min(rightRadius, h / 2, w / 2));

      ctx.moveTo(x + rLeft, y);
      ctx.lineTo(x + w - rRight, y);
      if (rRight > 0) {
        ctx.arcTo(x + w, y, x + w, y + rRight, rRight);
      } else {
        ctx.lineTo(x + w, y);
      }
      ctx.lineTo(x + w, y + h - rRight);
      if (rRight > 0) {
        ctx.arcTo(x + w, y + h, x + w - rRight, y + h, rRight);
      } else {
        ctx.lineTo(x + w, y + h);
      }
      ctx.lineTo(x + rLeft, y + h);
      if (rLeft > 0) {
        ctx.arcTo(x, y + h, x, y + h - rLeft, rLeft);
      } else {
        ctx.lineTo(x, y + h);
      }
      ctx.lineTo(x, y + rLeft);
      if (rLeft > 0) {
        ctx.arcTo(x, y, x + rLeft, y, rLeft);
      } else {
        ctx.lineTo(x, y);
      }
    }
  }
  if (drawFill) {
    ctx.fill();
  }
  if (drawBorder) {
    ctx.stroke();
  }
  ctx.restore();
}

export function drawKeyframe(
  ctx: CanvasRenderingContext2D,
  keyframe: TimelineKeyframe,
  coords: TimelineCoordinates,
  trackY: number,
  trackHeight: number,
  trackColor: [number, number, number],
  theme: TimelineTheme,
  width: number,
): void {
  const keyframeX =
    coords.timelineStartX +
    (keyframe.frame - coords.viewStartFrame) * coords.pixelRange;

  if (keyframeX < 0 - DRAW_BUFFER || keyframeX > width + DRAW_BUFFER) {
    return;
  }

  const keyframeY = trackY + trackHeight / 2;
  const opacity = keyframe.opacity !== undefined ? keyframe.opacity : 1.0;

  ctx.save();
  ctx.fillStyle = `rgba(${trackColor[0]}, ${trackColor[1]}, ${trackColor[2]}, ${opacity})`;
  ctx.strokeStyle = theme.keyframeBorderColor;
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  ctx.arc(keyframeX, keyframeY, KEYFRAME_RADIUS, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

export function drawFrameLabels(
  ctx: CanvasRenderingContext2D,
  coords: TimelineCoordinates,
  width: number,
  height: number,
  drawTicks: boolean = false,
  theme: TimelineTheme,
): void {
  ctx.fillStyle = theme.frameLabelColor;
  ctx.font = "10px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  if (drawTicks) {
    ctx.strokeStyle = theme.frameTickColor;
    ctx.lineWidth = 1;
  }

  for (
    let frame =
      Math.floor(coords.firstVisibleFrame / coords.frameDelta) * coords.frameDelta;
    frame <= coords.viewEndFrame;
    frame += coords.frameDelta
  ) {
    const frameX =
      coords.timelineStartX +
      (frame - coords.viewStartFrame) * coords.pixelRange;

    if (frameX >= -50 && frameX <= width + 50) {
      ctx.fillText(String(frame), frameX, FRAME_LABELS_HEIGHT / 2);
    }

    if (drawTicks) {
      ctx.beginPath();
      ctx.moveTo(frameX, FRAME_LABELS_HEIGHT);
      ctx.lineTo(frameX, height);
      ctx.stroke();
    }
  }
}

export function drawEndFrameLine(
  ctx: CanvasRenderingContext2D,
  coords: TimelineCoordinates,
  zoomLevel: number,
  width: number,
  height: number,
  theme: TimelineTheme,
): void {
  if (zoomLevel <= 1.000001) {
    return;
  }

  const endFrameX =
    coords.timelineStartX +
    (coords.endFrame - coords.viewStartFrame) * coords.pixelRange;

  if (endFrameX >= coords.timelineStartX && endFrameX <= width) {
    ctx.save();
    ctx.strokeStyle = theme.frameTickColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(endFrameX, FRAME_LABELS_HEIGHT);
    ctx.lineTo(endFrameX, height);
    ctx.stroke();
    ctx.restore();
  }
}

export function drawLabelOverlay(
  ctx: CanvasRenderingContext2D,
  coords: TimelineCoordinates,
  tracks: TimelineTrack[],
  hasPrompts: boolean,
  width: number,
  height: number,
  theme: TimelineTheme,
): void {
  ctx.fillStyle = theme.backgroundColor;
  ctx.fillRect(0, FRAME_LABELS_HEIGHT, TRACK_LABEL_WIDTH, height - FRAME_LABELS_HEIGHT);

  let currentTrackY = FRAME_LABELS_HEIGHT;

  if (hasPrompts) {
    ctx.fillStyle = theme.trackLabelColor;
    ctx.font = "12px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("Prompts", TRACK_LABEL_WIDTH - 10, currentTrackY + TRACK_HEIGHT / 2);

    ctx.strokeStyle = theme.trackSeparatorColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, currentTrackY);
    ctx.lineTo(TRACK_LABEL_WIDTH, currentTrackY);
    ctx.stroke();

    currentTrackY += TRACK_HEIGHT;
  }

  tracks.forEach((track: TimelineTrack) => {
    const trackHeight = TRACK_HEIGHT * (track.height_scale || 1.0);
    ctx.fillStyle = theme.trackLabelColor;
    ctx.font = "11px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(track.name, TRACK_LABEL_WIDTH - 10, currentTrackY + trackHeight / 2);

    ctx.strokeStyle = theme.trackSeparatorColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, currentTrackY);
    ctx.lineTo(TRACK_LABEL_WIDTH, currentTrackY);
    ctx.stroke();

    currentTrackY += trackHeight;
  });

  ctx.strokeStyle = theme.trackSeparatorColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(TRACK_LABEL_WIDTH, FRAME_LABELS_HEIGHT);
  ctx.lineTo(TRACK_LABEL_WIDTH, height);
  ctx.stroke();

  ctx.save();
  const shadowGradient = ctx.createLinearGradient(
    0,
    FRAME_LABELS_HEIGHT,
    0,
    FRAME_LABELS_HEIGHT + 8,
  );
  shadowGradient.addColorStop(0, theme.headerShadowColor);
  shadowGradient.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = shadowGradient;
  ctx.fillRect(0, FRAME_LABELS_HEIGHT, TRACK_LABEL_WIDTH, 8);
  ctx.restore();
}
