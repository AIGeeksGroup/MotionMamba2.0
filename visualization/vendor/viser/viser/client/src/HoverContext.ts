import React from "react";

// Extended hover context to include instanceId for instanced meshes and clickable state
export interface HoverState {
  isHovered: boolean;
  instanceId: number | null;
}

export const HoverableContext = React.createContext<{
  state: React.MutableRefObject<HoverState>;
  clickable: boolean;
  /** When set, all nodes with the same value highlight together when any is hovered. */
  highlightGroup?: string;
} | null>(null);
