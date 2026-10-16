// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import * as THREE from "three";
import { Message } from "./WebsocketMessages";
import { ViewerContext, ViewerContextContents } from "./ViewerContext";

/** Easier, hook version of makeThrottledMessageSender. */
export function useThrottledMessageSender(throttleMilliseconds: number) {
  const viewer = React.useContext(ViewerContext)!;
  return makeThrottledMessageSender(viewer, throttleMilliseconds);
}

/** Returns a function for sending messages, with automatic throttling. */
export function makeThrottledMessageSender(
  viewer: ViewerContextContents,
  throttleMilliseconds: number,
) {
  let readyToSend = true;
  let stale = false;

  // We coalesce messages by a "throttle key" so that rapid updates for different entities
  // (e.g. many `TimelinePromptMoveMessage`s for different prompts) don't overwrite each other.
  // Without this, only the last message within the window would be sent.
  const latestMessages = new Map<string, Message>();

  function getThrottleKey(message: Message): string {
    const anyMsg = message as any;
    const type = (anyMsg?.type as string) ?? "unknown";
    // Prefer common id fields so we coalesce per-entity, not globally per type.
    const preferredIdFields = [
      "prompt_id",
      "interval_id",
      "keyframe_id",
      "handle_id",
      "node_id",
      "scene_node_id",
      "camera_id",
      "material_id",
    ];
    for (const k of preferredIdFields) {
      if (typeof anyMsg?.[k] === "string") return `${type}:${k}:${anyMsg[k]}`;
    }
    // Fallback: first string field ending with `_id`.
    for (const k of Object.keys(anyMsg ?? {})) {
      if (k.endsWith("_id") && typeof anyMsg[k] === "string") return `${type}:${k}:${anyMsg[k]}`;
    }
    // Last resort: coalesce by message type.
    return type;
  }

  function sendQueued() {
    const viewerMutable = viewer.mutable.current;
    if (viewerMutable.sendMessage === null) return;
    if (latestMessages.size === 0) return;
    for (const msg of latestMessages.values()) {
      viewerMutable.sendMessage(msg);
    }
    latestMessages.clear();
    stale = false;
  }

  function send(message: Message) {
    const key = getThrottleKey(message);
    latestMessages.set(key, message);
    if (readyToSend) {
      sendQueued();
      readyToSend = false;

      setTimeout(() => {
        readyToSend = true;
        if (!stale) return;
        sendQueued();
      }, throttleMilliseconds);
    } else {
      stale = true;
    }
  }
  function flush() {
    const viewerMutable = viewer.mutable.current;
    if (viewerMutable.sendMessage === null) return;
    // Send all queued messages immediately.
    for (const msg of latestMessages.values()) {
      viewer.mutable.current.sendMessage(msg);
    }
    latestMessages.clear();
    stale = false;
  }
  return { send, flush };
}

/** Type guard for threejs textures. Meant to be used with `scene.background`. */
export function isTexture(
  background:
    | THREE.Color
    | THREE.Texture
    | THREE.CubeTexture
    | null
    | undefined,
): background is THREE.Texture {
  return (
    background !== null &&
    background !== undefined &&
    (background as THREE.Texture).isTexture !== undefined
  );
}
