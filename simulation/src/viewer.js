import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { smoothSurfaceNormals } from './surface.js';
import URDFLoader from '../vendor/urdf-loader/src/URDFLoader.js';

export const JOINTS = [
  'left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint',
  'left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint',
  'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint',
  'right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
  'waist_yaw_joint','waist_roll_joint','waist_pitch_joint',
  'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint',
  'left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',
  'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint',
  'right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint',
];

export class ReplayViewer {
  constructor(canvas, {mode='viser', showcase=false, transparent=false} = {}) {
    this.canvas = canvas;
    this.mode = mode;
    this.showcase = showcase;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias:true,
      alpha:transparent,
      powerPreference:'high-performance'
    });
    const gl = this.renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    this.software = /swiftshader|llvmpipe|software/i.test(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '');
    this.renderer.setPixelRatio(this.software ? .8 : Math.min(devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.renderer.shadowMap.enabled = !this.software;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    if (transparent) this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.scene.background = transparent ? null : new THREE.Color('#f3f4f7');
    this.scene.fog = transparent ? null : new THREE.Fog('#f3f4f7', 10, 35);
    this.camera = new THREE.PerspectiveCamera(35, 1, .02, 120);
    this.camera.up.set(0,0,1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .09;
    this.controls.minDistance = 1.1;
    this.controls.maxDistance = 18;
    this.controls.maxPolarAngle = Math.PI * .49;
    this.controls.enabled = !showcase;
    this.root = new THREE.Vector3();
    this.previous = new THREE.Vector3();
    this.cameraRoot = new THREE.Vector3();
    this.previousCameraRoot = new THREE.Vector3();
    this.currentSeconds = 0;
    this.hasPose = false;
    this.q1 = new THREE.Quaternion();
    this.q2 = new THREE.Quaternion();
    this.pose = new Float32Array(36);
    this.addEnvironment();
    this.resizeFrame = 0;
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
      cancelAnimationFrame(this.resizeFrame);
      this.resizeFrame = requestAnimationFrame(() => {
        if (this.data) {
          this.setTime(this.currentSeconds);
          this.render();
        }
      });
    });
    this.resizeObserver.observe(canvas.parentElement);
    this.resize();
  }

  addEnvironment() {
    // Bake studio reflections once; the replay remains entirely self-contained.
    const studio = new RoomEnvironment();
    studio.rotation.x = Math.PI / 2; // RoomEnvironment is Y-up; the G1 scene is Z-up.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.environment = pmrem.fromScene(studio, .04);
    this.scene.environment = this.environment.texture;
    studio.dispose();
    pmrem.dispose();
    this.hemisphere = new THREE.HemisphereLight(0xffffff,0x85898b,.55);
    this.scene.add(this.hemisphere);
    const sun = new THREE.DirectionalLight(0xfffaf4, 2);
    sun.position.set(4,-5,8);
    sun.castShadow = !this.software;
    sun.shadow.mapSize.set(1024,1024);
    Object.assign(sun.shadow.camera,{near:.1,far:25,left:-5,right:5,top:5,bottom:-5});
    sun.shadow.camera.updateProjectionMatrix();
    sun.shadow.normalBias = .008;
    sun.shadow.bias = -.0002;
    this.sun = sun;
    this.scene.add(sun,sun.target);
    const fill = new THREE.DirectionalLight(0xe7eeff,.6);
    fill.position.set(-3,2,4);
    const rim = new THREE.DirectionalLight(0xffffff,1.1);
    rim.position.set(-1,4,5);
    this.scene.add(fill,fill.target,rim,rim.target);
    this.studioLights = [[sun,4,-5,8],[fill,-3,2,4],[rim,-1,4,5]];

    // MuJoCo/SONIC uses the muted two-tone green checkerboard seen in the
    // reference simulator, with one metre cells and no bright grid overlay.
    const checkerCanvas = document.createElement('canvas');
    checkerCanvas.width = checkerCanvas.height = 256;
    const checker = checkerCanvas.getContext('2d');
    checker.fillStyle = '#51645b';
    checker.fillRect(0,0,256,256);
    checker.fillStyle = '#82968b';
    checker.fillRect(0,0,128,128);
    checker.fillRect(128,128,128,128);
    this.sonicFloorTexture = new THREE.CanvasTexture(checkerCanvas);
    this.sonicFloorTexture.colorSpace = THREE.SRGBColorSpace;
    this.sonicFloorTexture.wrapS = this.sonicFloorTexture.wrapT = THREE.RepeatWrapping;
    this.sonicFloorTexture.repeat.set(80,80);
    this.sonicFloorTexture.anisotropy = Math.min(8,this.renderer.capabilities.getMaxAnisotropy());

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(160,160),
      new THREE.MeshStandardMaterial({color:0xf3f4f7,roughness:1,metalness:0,envMapIntensity:.18}));
    floor.position.z = -.008;
    floor.receiveShadow = true;
    this.floor = floor;
    this.scene.add(floor);
    const sonicFloor = new THREE.Mesh(new THREE.PlaneGeometry(160,160),
      new THREE.MeshStandardMaterial({color:0xffffff,map:this.sonicFloorTexture,roughness:1,metalness:0,envMapIntensity:.12}));
    sonicFloor.position.z = -.008;
    sonicFloor.receiveShadow = true;
    sonicFloor.visible = false;
    this.sonicFloor = sonicFloor;
    this.scene.add(sonicFloor);
    // A world-space, anti-aliased grid fades away instead of meeting a hard horizon.
    const grid = new THREE.Mesh(new THREE.PlaneGeometry(160,160),new THREE.ShaderMaterial({
      transparent:true,depthWrite:false,toneMapped:false,
      uniforms:{center:{value:new THREE.Vector2()},color:{value:new THREE.Color('#c4c9d4')},opacity:{value:.36}},
      vertexShader:`varying vec2 worldXY;
        void main(){vec4 p=modelMatrix*vec4(position,1.0);worldXY=p.xy;gl_Position=projectionMatrix*viewMatrix*p;}`,
      fragmentShader:`varying vec2 worldXY;
        uniform vec2 center;uniform vec3 color;uniform float opacity;
        void main(){
          vec2 width=max(fwidth(worldXY),vec2(.0001));
          vec2 lines=abs(fract(worldXY-.5)-.5)/width;
          float line=1.0-min(min(lines.x,lines.y),1.0);
          float fade=1.0-smoothstep(1.5,8.0,distance(worldXY,center));
          gl_FragColor=vec4(color,line*fade*opacity);
        }`,
    }));
    grid.position.z = -.005;
    this.grid = grid;
    this.scene.add(grid);
    // Soft contact remains visible on clients where realtime shadows are disabled.
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(64,64,0,64,64,64);
    gradient.addColorStop(0,'rgba(0,0,0,.65)');
    gradient.addColorStop(.35,'rgba(0,0,0,.3)');
    gradient.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle = gradient;ctx.fillRect(0,0,128,128);
    this.contactTexture = new THREE.CanvasTexture(canvas);
    this.contact = new THREE.Mesh(new THREE.PlaneGeometry(1.35,1.1),new THREE.MeshBasicMaterial({
      map:this.contactTexture,transparent:true,opacity:.2,depthWrite:false,toneMapped:false,
    }));
    this.contact.position.z = -.003;
    this.scene.add(this.contact);
    // Interaction overlays are kept in world coordinates and use the same
    // clock as both reference and measured SONIC tracks.
    this.interactionGroup = new THREE.Group();
    this.scene.add(this.interactionGroup);
    this.interactionMarkers = [];
    this.interactionActive = null;
  }

  async loadRobot(progress) {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    const base = new URL('../assets/robot/',import.meta.url).href;
    loader.packages = {robot_description:base};
    loader.parseCollision = false;
    loader.parseVisual = true;
    await new Promise((resolve,reject) => {
      manager.onProgress = (_url,loaded,total) => progress(`Loading G1 meshes · ${loaded} / ${total}`);
      manager.onError = url => reject(new Error(`Unable to load robot mesh: ${url}`));
      manager.onLoad = resolve;
      loader.load(new URL('urdf/g1/main.urdf',base).href,robot => {
        this.robot = robot;
        this.scene.add(robot);
      },undefined,reject);
    });
    if (!this.robot) throw Error('G1 robot description did not load.');
    // Mesh callbacks can finish one task behind LoadingManager when a page is
    // restored or refreshed from cache. Wait for the visual tree to settle so
    // the MuJoCo copy never captures an empty URDF hierarchy.
    await this.waitForRobotVisuals(this.robot,progress);
    this.applyRobotMaterials();
    this.actualRobot = await this.cloneCompleteRobot(this.robot);
    this.actualRobot.visible = false;
    this.scene.add(this.actualRobot);
    // These are recorded poses, not commands. Preserve measured excursions
    // beyond URDF limits instead of silently changing either replay track.
    for (const robot of [this.robot,this.actualRobot])
      for (const joint of Object.values(robot.joints)) joint.ignoreLimits = true;
    const missing = JOINTS.filter(name => !this.robot.joints[name]);
    if (missing.length) throw Error(`Missing G1 joints: ${missing.join(', ')}`);
  }

  countRobotMeshes(robot) {
    let count = 0;
    robot?.traverse(object => {if (object.isMesh) count++;});
    return count;
  }

  async waitForRobotVisuals(robot,progress) {
    const timeout = performance.now()+10000;
    let previous = -1, stableSince = performance.now();
    while (performance.now()<timeout) {
      const count = this.countRobotMeshes(robot);
      if (count!==previous) {
        previous=count;
        stableSince=performance.now();
      }
      if (count>0 && performance.now()-stableSince>=400) return count;
      progress?.(count ? `Preparing G1 visuals · ${count} meshes` : 'Preparing G1 visuals…');
      await new Promise(resolve=>setTimeout(resolve,50));
    }
    throw Error('G1 visual meshes did not finish loading.');
  }

  async cloneCompleteRobot(robot) {
    for (let attempt=0;attempt<4;attempt++) {
      const expected=this.countRobotMeshes(robot);
      const clone=robot.clone(true);
      // Allow any cached loader callback queued in the current task to land,
      // then compare both complete visual trees before accepting the copy.
      await new Promise(resolve=>setTimeout(resolve,50));
      if (expected>0 && expected===this.countRobotMeshes(robot) &&
          expected===this.countRobotMeshes(clone)) return clone;
      await this.waitForRobotVisuals(robot);
    }
    throw Error('Unable to prepare the MuJoCo robot visuals.');
  }

  applyRobotMaterials() {
    const materials = new Map(), geometries = new WeakMap(), originals = new Set();
    const surfaces = {
      white: {color:0xb3b5b8,roughness:.43,metalness:.12},
      dark: {color:0x373c43,roughness:.42,metalness:.08},
      black: {color:0x20242a,roughness:.5,metalness:.04},
      visor: {color:0x171c23,roughness:.23,metalness:.12},
    };
    this.robot.traverse(object => {
      if (!object.isMesh) return;
      object.castShadow = true;
      object.receiveShadow = true;
      // Smooth curved STL surfaces while preserving creases and exact vertices.
      const geometry = object.geometry;
      if (!geometries.has(geometry)) geometries.set(geometry,smoothSurfaceNormals(geometry));
      object.geometry = geometries.get(geometry);
      let link = object.parent;
      while (link && !link.isURDFLink) link = link.parent;
      const convert = source => {
        const profile = link?.name === 'head_link' ? 'visor' : source.name;
        const key = `${source.uuid}:${profile}`;
        if (!materials.has(key)) {
          const material = new THREE.MeshStandardMaterial({
            color:source.color, map:source.map, side:source.side,
            transparent:source.transparent, opacity:source.opacity,
            depthWrite:source.depthWrite, roughness:.45, metalness:.08,
            envMapIntensity:.8, ...surfaces[profile],
          });
          material.name = `G1 ${profile || 'surface'}`;
          materials.set(key,material);
        }
        originals.add(source);
        return materials.get(key);
      };
      object.material = Array.isArray(object.material) ? object.material.map(convert) : convert(object.material);
    });
    originals.forEach(material => material.dispose());
  }

  setMotion(data, clip, executionData=null) {
    if (data.length !== clip.frames*36) throw Error('The motion file has an invalid frame count.');
    this.data = data;
    this.clip = clip;
    if (executionData && executionData.length!==clip.frames*36) throw Error('The simulation recording has an invalid frame count.');
    this.executionData = executionData;
    this.buildInteractionOverlay(clip);
    this.hasPose = false;
    this.setTime(0);
    this.resetCamera();
  }

  // Exported Viser sessions store root points as ARDY ground-plane [x, 0, z]
  // values. The converter records MuJoCo ground coordinates as [z, x], so
  // apply the same axis permutation used by the replay qpos.
  buildInteractionOverlay(clip) {
    if (!this.interactionGroup) return;
    this.interactionGroup.clear();
    this.interactionMarkers = [];
    this.interactionActive = null;
    const points = [];
    const add = (frame, xz, kind='constraint', heading=null) => {
      if (!Array.isArray(xz) || xz.length < 2 || !xz.every(Number.isFinite)) return;
      const item = {frame:Math.max(0,Math.min(clip.frames-1,Number(frame)||0)), x:Number(xz[1]), y:Number(xz[0]), kind, heading};
      const duplicate = points.find(p => p.frame===item.frame && Math.hypot(p.x-item.x,p.y-item.y)<1e-4);
      if (!duplicate) points.push(item);
    };
    const interaction = clip.interaction || {};
    for (const accepted of interaction.accepted_control_timeline || []) {
      const start = Number(accepted.start_frame || 0);
      for (const waypoint of accepted.control?.root2d || [])
        add(start + Number(waypoint.frame || 0), waypoint.xz, waypoint.origin === 'keyboard_preview' ? 'keyboard' : 'constraint', waypoint.heading);
    }
    const keyframes = clip.constraints?.['2D Root']?.keyframes || {};
    for (const [frame, value] of Object.entries(keyframes))
      add(Number(frame), value.position ? [value.position[0], value.position[2]] : null, value.origin === 'keyboard_preview' ? 'keyboard' : 'mouse', value.heading);
    points.sort((a,b)=>a.frame-b.frame);
    this.interactionPoints = points;
    if (!points.length) return;
    const guideGeometry = new THREE.BufferGeometry().setFromPoints(points.map(point=>new THREE.Vector3(point.x,point.y,.018)));
    const guidePath = new THREE.Line(guideGeometry,new THREE.LineBasicMaterial({color:0xf08a42,transparent:true,opacity:.82,depthWrite:false}));
    this.interactionGroup.add(guidePath);
    const materials = {
      constraint:new THREE.MeshBasicMaterial({color:0xf08a42,transparent:true,opacity:.8,depthWrite:false}),
      keyboard:new THREE.MeshBasicMaterial({color:0x6b8cff,transparent:true,opacity:.9,depthWrite:false}),
      mouse:new THREE.MeshBasicMaterial({color:0xf08a42,transparent:true,opacity:.9,depthWrite:false}),
    };
    const geometryMarker = new THREE.SphereGeometry(.055,12,8);
    for (const point of points) {
      const marker = new THREE.Mesh(geometryMarker,materials[point.kind] || materials.constraint);
      marker.position.set(point.x,point.y,.035);
      marker.userData.interaction = point;
      this.interactionGroup.add(marker);
      this.interactionMarkers.push(marker);
    }
    this.interactionActive = new THREE.Mesh(new THREE.RingGeometry(.075,.105,24),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.95,side:THREE.DoubleSide,depthWrite:false}));
    this.interactionActive.position.z=.04;
    this.interactionActive.visible=false;
    this.interactionGroup.add(this.interactionActive);
  }

  applyPose(robot, data, seconds) {
    const f = THREE.MathUtils.clamp(seconds*this.clip.fps,0,this.clip.frames-1);
    const a = Math.floor(f), b = Math.min(a+1,this.clip.frames-1), t = f-a;
    for(let j=0;j<3;j++) this.pose[j]=THREE.MathUtils.lerp(data[a*36+j],data[b*36+j],t);
    this.q1.set(data[a*36+4],data[a*36+5],data[a*36+6],data[a*36+3]).normalize();
    this.q2.set(data[b*36+4],data[b*36+5],data[b*36+6],data[b*36+3]).normalize();
    this.q1.slerp(this.q2,t);
    robot.position.fromArray(this.pose);
    robot.quaternion.copy(this.q1);
    for(let j=0;j<JOINTS.length;j++) {
      const x=data[a*36+7+j], y=data[b*36+7+j];
      const delta=Math.atan2(Math.sin(y-x),Math.cos(y-x));
      this.pose[7+j] = x+delta*t;
      robot.joints[JOINTS[j]].setJointValue(x+delta*t);
    }
  }

  setTime(seconds) {
    if(!this.robot || !this.data) return;
    this.currentSeconds = seconds;
    this.applyPose(this.robot,this.data,seconds);
    if(this.executionData)this.applyPose(this.actualRobot,this.executionData,seconds);
    this.root.copy(this.robot.position);
    this.cameraRoot.copy(this.robot.position);
    if (this.mode === 'sonic' && this.executionData) this.cameraRoot.copy(this.actualRobot.position);
    else if (this.mode === 'split' && this.executionData) this.cameraRoot.add(this.actualRobot.position).multiplyScalar(.5);
    if(this.hasPose) {
      const dx=this.cameraRoot.x-this.previousCameraRoot.x, dy=this.cameraRoot.y-this.previousCameraRoot.y;
      this.camera.position.x+=dx;this.camera.position.y+=dy;
      this.controls.target.x+=dx;this.controls.target.y+=dy;
    }
    this.previous.copy(this.root);
    this.previousCameraRoot.copy(this.cameraRoot);
    this.hasPose=true;
    this.grid.material.uniforms.center.value.set(this.root.x,this.root.y);
    this.contact.position.set(this.root.x,this.root.y,-.003);
    this.contact.material.opacity = .24 * Math.max(.1,1 - Math.max(0,this.root.z-.8)*1.4);
    if (this.interactionPoints?.length && this.interactionActive) {
      const frame = seconds * this.clip.fps;
      let active = null;
      for (const point of this.interactionPoints) if (point.frame <= frame) active = point;
      this.interactionActive.visible = !!active;
      if (active) this.interactionActive.position.set(active.x,active.y,.04);
      for (const marker of this.interactionMarkers) {
        const current = marker.userData.interaction === active;
        marker.scale.setScalar(current ? 1.45 : 1);
        marker.material.opacity = current ? 1 : .72;
      }
    }
    for (const [light,x,y,z] of this.studioLights) {
      light.position.set(this.root.x+x,this.root.y+y,z);
      light.target.position.set(this.root.x,this.root.y,.7);
    }
  }

  resetCamera() {
    const {x,y}=this.cameraRoot;
    const narrow = this.mode === 'split';
    this.camera.position.set(x+2.25,y-3.35,narrow ? 1.8 : 1.9);
    this.controls.target.set(x,y,.77);
    this.controls.update();
  }

  setViewMode(mode) {
    if (!['viser','sonic','split'].includes(mode)) throw Error(`Unknown view mode: ${mode}`);
    this.mode = mode;
    if (this.data) {
      this.hasPose = false;
      this.setTime(this.currentSeconds);
      this.resetCamera();
    }
    this.resize();
  }

  applyStyle(style) {
    if (this.activeStyle === style) return;
    this.activeStyle = style;
    const sonic = style === 'sonic';
    this.scene.background.set(sonic ? '#15221c' : '#f3f4f7');
    this.scene.fog.color.copy(this.scene.background);
    this.floor.visible = !sonic;
    this.sonicFloor.visible = sonic;
    this.floor.material.color.set('#f0f1f5');
    this.grid.material.uniforms.color.value.set('#b4bccb');
    this.grid.material.uniforms.opacity.value = sonic ? 0 : .28;
    this.hemisphere.intensity = sonic ? .75 : .55;
    this.contact.visible = !sonic;
  }

  resize() {
    const {width,height}=this.canvas.parentElement.getBoundingClientRect();
    this.width = Math.max(width,1);
    this.height = Math.max(height,1);
    this.renderer.setSize(this.width,this.height,false);
    this.camera.aspect=this.width/(this.mode === 'split' ? 2 : 1)/this.height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    const {width,height} = this;
    if (this.mode === 'split') {
      // Independent reference/physics poses share one timestamp, camera and renderer.
      const half = width/2;
      this.renderer.setScissorTest(true);
      this.renderer.setViewport(0,0,half,height);
      this.renderer.setScissor(0,0,half,height);
      this.applyStyle('viser');
      this.selectRobot(false);
      this.renderer.render(this.scene,this.camera);
      this.renderer.setViewport(half,0,width-half,height);
      this.renderer.setScissor(half,0,width-half,height);
      this.applyStyle('sonic');
      this.selectRobot(!!this.executionData);
      this.renderer.render(this.scene,this.camera);
      this.renderer.setScissorTest(false);
    } else {
      this.renderer.setViewport(0,0,width,height);
      this.applyStyle(this.mode);
      this.selectRobot(this.mode==='sonic' && !!this.executionData);
      this.renderer.render(this.scene,this.camera);
    }
  }

  selectRobot(actual) {
    this.robot.visible = !actual;
    this.actualRobot.visible = actual;
    (actual ? this.actualRobot : this.robot).updateMatrixWorld(true);
  }

  poseStatus() {
    const pose = robot => robot ? [...robot.position.toArray(),...robot.quaternion.toArray(),
      ...JOINTS.map(name=>robot.joints[name].jointValue[0])] : null;
    return {reference:pose(this.robot),actual:this.executionData ? pose(this.actualRobot) : null};
  }

  dispose() {
    this.resizeObserver.disconnect();
    cancelAnimationFrame(this.resizeFrame);
    this.controls.dispose();
    this.scene.traverse(object => {
      object.geometry?.dispose();
      const materials=Array.isArray(object.material)?object.material:[object.material];
      for(const m of materials) m?.dispose();
    });
    this.renderer.dispose();
    this.environment.dispose();
    this.contactTexture.dispose();
    this.sonicFloorTexture.dispose();
  }
}
