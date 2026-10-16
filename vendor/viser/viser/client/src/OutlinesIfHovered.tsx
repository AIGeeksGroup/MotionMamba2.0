import React from "react";
import { useFrame } from "@react-three/fiber";
import { HoverableContext } from "./HoverContext";
import { Outlines } from "./Outlines";
import * as THREE from "three";
import { ViewerContext } from "./ViewerContext";

/** Outlines object, which should be placed as a child of all meshes that might
 * be clickable. */
export function OutlinesIfHovered(
  props: {
    unmountOnHide?: boolean;
    enableCreaseAngle?: boolean;
  } = {
    unmountOnHide: false, // Useful when outlines are combined with <Instances />.
    enableCreaseAngle: false,
  },
) {
  const hoverContext = React.useContext(HoverableContext);
  if (hoverContext === null || !hoverContext.clickable) return null;
  return <_OutlinesIfHovered {...props} />;
}

function _OutlinesIfHovered(props: {
  unmountOnHide?: boolean;
  enableCreaseAngle?: boolean;
}) {
  const groupRef = React.useRef<THREE.Group>(null);
  const hoverContext = React.useContext(HoverableContext);
  const viewer = React.useContext(ViewerContext);
  const [mounted, setMounted] = React.useState(true);

  const creaseAngle = props.enableCreaseAngle ? Math.PI : 0.0;

  useFrame(() => {
    if (hoverContext === null || !hoverContext.clickable) return;
    const isGroupHovered =
      hoverContext.highlightGroup &&
      viewer?.mutable.current.hoveredHighlightGroups.has(hoverContext.highlightGroup);
    const showOutline =
      hoverContext.state.current.isHovered || Boolean(isGroupHovered);
    if (props.unmountOnHide) {
      if (mounted !== showOutline) setMounted(showOutline);
      return;
    }
    if (groupRef.current !== null) groupRef.current.visible = showOutline;
  });

  return !mounted ? null : (
    <Outlines
      ref={groupRef}
      thickness={10}
      screenspace={true}
      color={0xfbff00}
      opacity={0.8}
      transparent={true}
      angle={creaseAngle}
    />
  );
}
