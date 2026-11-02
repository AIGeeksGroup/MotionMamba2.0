import * as THREE from 'three';
import { ReplayViewer } from './viewer.js?v=2';

const stage = document.getElementById('overview-stage');
const canvas = document.getElementById('overview-canvas');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

let viewer = null;
let actors = [];
let ready = false;
let visible = false;
let initializing = null;
let elapsed = 0;
let lastFrame = performance.now();
let animationFrame = 0;
const facingAxis = new THREE.Vector3(1, 0, 0);
const upAxis = new THREE.Vector3(0, 0, 1);
const headingCorrection = new THREE.Quaternion();
const headingDirection = new THREE.Vector3();

async function loadManifest() {
  const response = await fetch(new URL('../manifest.json', import.meta.url));
  if (!response.ok) throw new Error('Overview manifest could not be loaded.');
  return response.json();
}

async function loadMotion(clip) {
  const response = await fetch(new URL(`../motions/${clip.file}`, import.meta.url));
  if (!response.ok) throw new Error(`Motion download failed (${response.status}).`);
  const data = new Float32Array(await response.arrayBuffer());
  if (data.length !== clip.frames * 36 || !data.every(Number.isFinite)) {
    throw new Error('Motion data is incomplete.');
  }
  return data;
}

function configureScene() {
  viewer.mode = 'viser';
  viewer.controls.enabled = false;
  viewer.contact.visible = true;
  viewer.interactionGroup.visible = false;
  viewer.scene.background = null;
  viewer.scene.fog = null;
  viewer.grid.visible = false;
  viewer.sonicFloor.visible = false;
  viewer.floor.visible = true;
  viewer.floor.material.dispose();
  viewer.floor.material = new THREE.ShadowMaterial({
    color: viewer.theme.ink,
    opacity: .14,
    transparent: true,
    depthWrite: false
  });

  viewer.camera.fov = 24;
  viewer.camera.near = .1;
  viewer.camera.far = 180;
  viewer.camera.position.set(0, -10.5, 5);
  viewer.controls.target.set(0, 0, .75);
  viewer.camera.updateProjectionMatrix();
  viewer.controls.update();

  viewer.sun.position.set(12, -16, 28);
  Object.assign(viewer.sun.shadow.camera, {
    near: .1,
    far: 70,
    left: -22,
    right: 22,
    top: 14,
    bottom: -14
  });
  viewer.sun.shadow.camera.updateProjectionMatrix();
}

async function initialize() {
  if (ready) return;
  if (initializing) return initializing;
  initializing = (async () => {
    const manifest = await loadManifest();
    const clips = manifest.scenarios.map((scenario) => scenario.clips[0]);
    const motions = await Promise.all(clips.map(loadMotion));

    viewer = new ReplayViewer(canvas, { mode: 'viser', showcase: true, transparent: true });
    await viewer.loadRobot(() => {});
    configureScene();

    const extraRobots = await Promise.all(
      Array.from({ length: 10 }, () => viewer.cloneCompleteRobot(viewer.robot))
    );
    const robots = [viewer.robot, viewer.actualRobot, ...extraRobots];
    const firstContact = viewer.contact;
    viewer.scene.remove(firstContact);

    actors = robots.map((robot, index) => {
      viewer.scene.remove(robot);
      robot.visible = true;
      Object.values(robot.joints).forEach((joint) => { joint.ignoreLimits = true; });

      const group = new THREE.Group();
      const column = index % 6;
      const row = Math.floor(index / 6);
      group.position.set((column - 2.5) * 2, (row - .5) * 2.4, 0);
      group.add(robot);

      const contact = index === 0 ? firstContact : firstContact.clone();
      contact.material = firstContact.material.clone();
      contact.visible = true;
      group.add(contact);
      viewer.scene.add(group);

      return {
        robot,
        group,
        contact,
        clip: clips[index],
        data: motions[index],
        phase: clips[index].duration * index / robots.length
      };
    });

    updateActors(0);
    viewer.resize();
    render();
    stage.dataset.actorCount = String(actors.length);
    stage.classList.add('is-ready');
    ready = true;
  })().catch((error) => {
    stage.classList.add('has-error');
    stage.dataset.error = error?.message || String(error);
    console.error(error);
  });
  return initializing;
}

function updateActors(time) {
  actors.forEach((actor) => {
    const localTime = (time + actor.phase) % actor.clip.duration;
    viewer.clip = actor.clip;
    viewer.applyPose(actor.robot, actor.data, localTime);
    actor.robot.position.x = 0;
    actor.robot.position.y = 0;

    headingDirection.copy(facingAxis).applyQuaternion(actor.robot.quaternion);
    const heading = Math.atan2(headingDirection.y, headingDirection.x);
    headingCorrection.setFromAxisAngle(upAxis, -Math.PI / 2 - heading);
    actor.robot.quaternion.premultiply(headingCorrection);

    actor.contact.position.set(actor.robot.position.x, actor.robot.position.y, -.003);
    actor.contact.material.opacity = .16 * Math.max(.1, 1 - Math.max(0, actor.robot.position.z - .8) * 1.4);
  });
}

function render() {
  if (!viewer || !ready && !actors.length) return;
  viewer.controls.update();
  viewer.renderer.setScissorTest(false);
  viewer.renderer.setViewport(0, 0, viewer.width, viewer.height);
  viewer.renderer.render(viewer.scene, viewer.camera);
}

function animate(now) {
  animationFrame = requestAnimationFrame(animate);
  const delta = Math.min((now - lastFrame) / 1000, .1);
  lastFrame = now;
  if (!ready || !visible || document.hidden) return;
  if (!reducedMotion) elapsed += delta;
  updateActors(elapsed);
  render();
}

const observer = new IntersectionObserver(([entry]) => {
  visible = entry.isIntersecting;
  lastFrame = performance.now();
  if (visible) initialize();
}, { rootMargin: '360px 0px', threshold: .01 });

observer.observe(stage);
animationFrame = requestAnimationFrame(animate);

window.addEventListener('pagehide', () => {
  observer.disconnect();
  cancelAnimationFrame(animationFrame);
  viewer?.dispose();
}, { once: true });
