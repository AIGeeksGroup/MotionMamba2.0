import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

// Display-only hand in the existing palm frame: +X along the fingers,
// Z across the palm. No joints or collision geometry are added to MuJoCo.
export function createFiveFingerHand(side) {
  const hand = new THREE.Group();
  hand.name = `${side}_five_finger_visual`;
  const mirror = side === "left" ? 1 : -1;
  const shell = new THREE.MeshStandardMaterial({
    color: 0xbfc7cb, metalness: 0.52, roughness: 0.38,
  });
  const rubber = new THREE.MeshStandardMaterial({
    color: 0x22272b, metalness: 0.12, roughness: 0.7,
  });
  const joint = new THREE.MeshStandardMaterial({
    color: 0x454e55, metalness: 0.65, roughness: 0.38,
  });

  function box(parent, size, position, material, radius = 0.003) {
    const mesh = new THREE.Mesh(new RoundedBoxGeometry(...size, 2, radius), material);
    mesh.position.set(...position);
    parent.add(mesh);
    return mesh;
  }

  box(hand, [0.066, 0.027, 0.074], [0.037, 0, 0], shell, 0.007);
  box(hand, [0.052, 0.004, 0.061], [0.039, -mirror * 0.014, 0], rubber, 0.002);
  box(hand, [0.02, 0.03, 0.039], [0.003, 0, 0], joint, 0.004);

  function finger(name, base, lengths, width, spread = 0, bend = 0.07) {
    const digit = new THREE.Group();
    digit.name = name;
    digit.position.set(...base);
    digit.rotation.y = spread;
    hand.add(digit);
    let parent = digit;
    for (const length of lengths) {
      const segment = new THREE.Group();
      segment.rotation.z = mirror * bend;
      parent.add(segment);
      const knuckle = new THREE.Mesh(
        new THREE.CylinderGeometry(width * 0.52, width * 0.52, width * 0.94, 12), joint,
      );
      knuckle.rotation.x = Math.PI / 2;
      segment.add(knuckle);
      box(segment, [length - 0.002, width, width], [length / 2, 0, 0], shell);
      box(segment, [length * 0.65, 0.0025, width * 0.8],
        [length * 0.56, -mirror * width * 0.52, 0], rubber, 0.001);
      const tip = new THREE.Group();
      tip.position.x = length;
      segment.add(tip);
      parent = tip;
    }
    return digit;
  }

  // The URDF mirrors left/right palms across Y, not Z. Index/little must
  // retain their Z ordering; the opposable thumb opens toward the palm side.
  finger("index", [0.069, 0, 0.027], [0.032, 0.022, 0.017], 0.013, -0.025);
  finger("middle", [0.071, 0, 0.009], [0.036, 0.025, 0.019], 0.014);
  finger("ring", [0.069, 0, -0.009], [0.033, 0.023, 0.017], 0.013, 0.02);
  finger("little", [0.062, 0, -0.027], [0.025, 0.018, 0.014], 0.011, 0.07);
  const thumb = finger("thumb", [0.0255, -mirror * 0.018, 0.032],
    [0.025, 0.020, 0.016], 0.014, -0.3, 0.04);
  thumb.rotation.z = -mirror * 0.85;

  // The visual fingers are rigid within the palm. Bake their transforms once
  // and submit one mesh per material instead of 48 draw calls per hand (plus
  // another shadow pass). The wrist attachment and visible shape are unchanged.
  hand.updateMatrixWorld(true);
  const groups = new Map();
  hand.traverse((object) => {
    if (!object.isMesh) return;
    const geometries = groups.get(object.material) || [];
    const baked = object.geometry.index ? object.geometry.toNonIndexed() : object.geometry.clone();
    geometries.push(baked.applyMatrix4(object.matrixWorld));
    groups.set(object.material, geometries);
    object.geometry.dispose();
  });
  hand.clear();
  for (const [material, geometries] of groups) {
    const geometry = mergeGeometries(geometries, false);
    for (const part of geometries) part.dispose();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true; mesh.receiveShadow = true;
    hand.add(mesh);
  }
  return hand;
}
