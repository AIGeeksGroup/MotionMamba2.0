// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "./App.css";
import "./index.css";

import { CameraControls } from "@react-three/drei";
import * as THREE from "three";
import React from "react";
import { UseSceneTree } from "./SceneTree";

import { UseGui } from "./ControlPanel/GuiState";
import { GetRenderRequestMessage, Message } from "./WebsocketMessages";

// Type definitions for all mutable state.
export type ViewerMutable = {
  // Function references.
  sendMessage: (message: Message) => void;
  sendCamera: (() => void) | null;
  resetCameraView: (() => void) | null;

  // DOM/Three.js references.
  canvas: HTMLCanvasElement | null;
  canvas2d: HTMLCanvasElement | null;
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  backgroundMaterial: THREE.ShaderMaterial | null;
  cameraControl: CameraControls | null;

  // Scene management.
  nodeRefFromName: {
    [name: string]: undefined | THREE.Object3D;
  };

  // Message and rendering state.
  messageQueue: Message[];
  getRenderRequestState: "ready" | "triggered" | "pause" | "in_progress";
  getRenderRequest: null | GetRenderRequestMessage;

  // Interaction state.
  scenePointerInfo: {
    enabled: false | "click" | "rect-select"; // Enable box events.
    dragStart: [number, number]; // First mouse position.
    dragEnd: [number, number]; // Final mouse position.
    isDragging: boolean;
  };

  // Skinned mesh state.
  skinnedMeshState: {
    [name: string]: {
      initialized: boolean;
      dirty: boolean; // Flag to track if bones need updating.
      poses: {
        wxyz: [number, number, number, number];
        position: [number, number, number];
      }[];
    };
  };

  // Global hover state tracking.
  hoveredElementsCount: number;
  /** When a node with highlight_group is hovered, that group id is added here so all nodes in the group show the outline. */
  hoveredHighlightGroups: Set<string>;

  /** Names of transform-control nodes currently being dragged. Pose is not applied from scene state while dragging to avoid jitter. */
  transformControlsDraggingNames: Set<string>;
};

export type ViewerContextContents = {
  // Non-mutable state.
  messageSource: "websocket" | "file_playback";

  // Zustand state hooks and actions.
  useSceneTree: UseSceneTree["store"];
  sceneTreeActions: UseSceneTree["actions"];
  useEnvironment: ReturnType<
    typeof import("./EnvironmentState").useEnvironmentState
  >;
  useGui: UseGui;
  useDevSettings: ReturnType<
    typeof import("./DevSettingsStore").useDevSettingsStore
  >;

  // Single reference to all mutable state.
  mutable: React.MutableRefObject<ViewerMutable>;

  // Keyboard listening state.
  keyboardListenEnabled: React.MutableRefObject<boolean>;
  
  // Camera keyboard controls enabled state (using state hook to trigger re-renders).
  cameraKeyboardControlsEnabled: {
    value: boolean;
    setValue: (value: boolean) => void;
  };
};

export const ViewerContext = React.createContext<null | ViewerContextContents>(
  null,
);
