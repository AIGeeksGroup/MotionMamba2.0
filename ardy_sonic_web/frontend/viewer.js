import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader from "/vendor/urdf-loader/src/URDFLoader.js";
import { createFiveFingerHand } from "/static/five-finger-hand.js?v=20261008-hand-frame2";
import { clusterStlGeometry } from "/static/visual-lod.mjs?v=20261008-runtime2";

export const G1_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
];

export class ActualStateViewer {
  constructor(canvas, onProgress = () => {}) {
    this.canvas = canvas;
    this.onProgress = onProgress;
    this.robot = null;
    this.handVisuals = [];
    this.handStyle = "five";
    this.ready = false;
    this.followCamera = true;
    this.previousRoot = new THREE.Vector3();
    this.currentRoot = new THREE.Vector3();
    this.hasRoot = false;
    this.lastRenderTime = performance.now();
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    this.renderMs = 0;
    this.slowFrames = 0;
    this.quality = 'balanced';
  }

  async init() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    const gl = this.renderer.getContext();
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    this.gpuName = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : '';
    if (/swiftshader|llvmpipe|software/i.test(this.gpuName)) {
      this.quality = 'performance'; this.pixelRatio = 0.75;
    }
    document.documentElement.dataset.renderQuality = this.quality;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.shadowMap.enabled = this.quality !== 'performance';
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.5;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x121a12);
    this.scene.fog = new THREE.FogExp2(0x121a12, 0.04);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(3.5, -3.5, 2.5);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0.9);
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 20;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.update();

    this.addEnvironment();
    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
    await this.loadRobot();
    this.ready = true;
  }

  addEnvironment() {
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(4, -2, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    Object.assign(key.shadow.camera, { near: 0.1, far: 30, left: -5, right: 5, top: 5, bottom: -5 });
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x80c0ff, 0.7);
    fill.position.set(-3, 4, 2);
    this.scene.add(fill);

    const back = new THREE.DirectionalLight(0xffffff, 0.5);
    back.position.set(-2, 2, 4);
    this.scene.add(back);
    this.scene.add(new THREE.HemisphereLight(0x6a8aaa, 0x2a3a20, 0.8));

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.MeshStandardMaterial({ color: 0x1a2820, roughness: 0.95 }),
    );
    floor.receiveShadow = true;
    floor.position.z = -0.003;
    this.scene.add(floor);

    const fineGrid = new THREE.GridHelper(20, 40, 0x4a8a5a, 0x3a6a48);
    fineGrid.rotation.x = Math.PI / 2;
    this.scene.add(fineGrid);
    const majorGrid = new THREE.GridHelper(20, 20, 0x6aaa7a, 0x6aaa7a);
    majorGrid.rotation.x = Math.PI / 2;
    majorGrid.position.z = 0.001;
    this.scene.add(majorGrid);
  }

  async loadRobot() {
    this.onProgress("Loading G1 robot description…");
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.packages = { robot_description: "/robot" };
    loader.parseCollision = false;

    await new Promise((resolve, reject) => {
      manager.onProgress = (_url, loaded, total) => {
        this.onProgress(`Loading G1 meshes (${loaded} / ${total})…`);
      };
      manager.onLoad = resolve;
      manager.onError = (url) => reject(new Error(`Failed to load robot asset: ${url}`));
      loader.load("/robot/urdf/g1/main.urdf", (robot) => {
        this.robot = robot;
        robot.traverse((object) => {
          if (!object.isMesh) return;
          object.castShadow = true;
          object.receiveShadow = true;
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          for (const material of materials) {
            if (!material) continue;
            material.metalness = material.color?.getHex() < 0x777777 ? 0.68 : 0.42;
            material.roughness = material.color?.getHex() < 0x777777 ? 0.34 : 0.52;
          }
        });
        this.scene.add(robot);
      }, undefined, reject);
    });

    const missing = G1_JOINTS.filter((name) => !this.robot.joints[name]);
    if (missing.length) throw new Error(`G1 URDF is missing joints: ${missing.join(", ")}`);
    for (const side of ["left", "right"]) {
      const palm = this.robot.links[`${side}_hand_palm_link`];
      if (!palm) throw new Error(`G1 URDF is missing the ${side} palm attachment`);
      const originalMeshes = [];
      palm.traverse((object) => {
        if (object.isMesh) originalMeshes.push({ mesh: object, visible: object.visible });
      });
      const fiveFinger = createFiveFingerHand(side);
      palm.add(fiveFinger);
      this.handVisuals.push({ originalMeshes, fiveFinger });
    }
    this.setHandStyle(this.handStyle);
    if (this.quality === 'performance') this.useLightweightMaterials();
    this.onProgress("Preparing actual-state stream…");
  }

  setHandStyle(style) {
    if (style !== "five" && style !== "original") return;
    this.handStyle = style;
    for (const { originalMeshes, fiveFinger } of this.handVisuals) {
      for (const { mesh, visible } of originalMeshes) {
        mesh.visible = style === "original" && visible;
      }
      fiveFinger.visible = style === "five";
    }
  }

  useLightweightMaterials() {
    const converted = new Map();
    this.scene.traverse((object) => {
      if (!object.isMesh) return;
      if (!object.geometry.userData.sonicVisualLod) {
        object.updateWorldMatrix(true,false);
        const scale=new THREE.Vector3().setFromMatrixScale(object.matrixWorld);
        const cell=(/swiftshader|llvmpipe|software/i.test(this.gpuName) ? .004 : .002)/Math.max(scale.x,scale.y,scale.z,1e-6);
        const lod=clusterStlGeometry(object.geometry,cell);
        if(lod) { object.geometry.dispose();object.geometry=lod; }
      }
      const convert = (material) => {
        if (!material?.isMeshStandardMaterial) return material;
        if (!converted.has(material)) {
          converted.set(material, new THREE.MeshLambertMaterial({
            color: material.color, map: material.map, side: material.side,
            transparent: material.transparent, opacity: material.opacity,
            vertexColors: material.vertexColors,
          }));
        }
        return converted.get(material);
      };
      object.material = Array.isArray(object.material) ? object.material.map(convert) : convert(object.material);
    });
    for (const material of converted.keys()) material.dispose();
  }

  setQpos(qpos) {
    if (!this.ready || !qpos || qpos.length !== 36) return;
    const root = this.currentRoot.set(qpos[0], qpos[1], qpos[2]);
    this.robot.position.copy(root);
    this.robot.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]).normalize();
    for (let index = 0; index < G1_JOINTS.length; index += 1) {
      this.robot.joints[G1_JOINTS[index]].setJointValue(qpos[index + 7]);
    }

    if (this.followCamera) {
      if (this.hasRoot) {
        const dx = root.x - this.previousRoot.x;
        const dy = root.y - this.previousRoot.y;
        this.camera.position.x += dx;
        this.camera.position.y += dy;
      }
      this.controls.target.set(root.x, root.y, Math.max(0.7, root.z + 0.1));
    }
    this.previousRoot.copy(root);
    this.hasRoot = true;
  }

  setFollow(enabled) {
    this.followCamera = enabled;
    if (enabled && this.hasRoot) {
      this.controls.target.set(this.previousRoot.x, this.previousRoot.y, Math.max(0.7, this.previousRoot.z + 0.1));
    }
  }

  render() {
    if (!this.renderer) return;
    const now = performance.now();
    const frameMs = now - this.lastRenderTime;
    const deltaTime = Math.min(0.1, Math.max(0, (now - this.lastRenderTime) / 1000));
    this.lastRenderTime = now;
    this.controls.update(deltaTime);
    const started = performance.now();
    this.renderer.render(this.scene, this.camera);
    this.renderMs = this.renderMs * 0.9 + (performance.now() - started) * 0.1;
    const slow = this.renderMs > 22 || (frameMs > 35 && document.visibilityState === 'visible');
    this.slowFrames = slow ? this.slowFrames + 1 : Math.max(0, this.slowFrames-1);
    if (this.slowFrames >= 45 && this.quality === 'balanced') {
      // Reduce display cost on software WebGL/slow client GPUs, independently
      // of the simulation. Keep the five-finger geometry and motion untouched.
      this.quality = 'performance'; this.pixelRatio = 1;
      document.documentElement.dataset.renderQuality = this.quality;
      this.renderer.shadowMap.enabled = false;
      this.useLightweightMaterials();
      this.resize();
      this.slowFrames = 0;
    } else if (this.slowFrames >= 45 && this.quality === 'performance' && this.pixelRatio > 0.5) {
      this.pixelRatio = 0.5; this.resize(); this.slowFrames = 0;
    }
  }

  performanceInfo() {
    return { quality: this.quality, pixelRatio: this.pixelRatio,
      renderMs: this.renderMs, drawCalls: this.renderer?.info.render.calls,
      triangles: this.renderer?.info.render.triangles, gpu: this.gpuName };
  }

  resize() {
    const width = this.canvas.clientWidth || window.innerWidth;
    const height = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
  }
}
