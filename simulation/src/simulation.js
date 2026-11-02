import { ReplayViewer } from './viewer.js';

const grid = document.getElementById('simulation-grid');
const switchButtons = [...document.querySelectorAll('.simulation-view-switch [data-simulation-view]')];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const motionBuffers = new Map();
const players = [];
let activeView = 'split';
let lastFrame = performance.now();

const promptColors = [
  'var(--prompt-blue)', 'var(--prompt-sky)', 'var(--prompt-cyan)',
  'var(--prompt-teal)', 'var(--prompt-sea)', 'var(--prompt-green)'
];
const promptAccents = [
  'var(--blue)', 'var(--sky)', 'var(--cyan)',
  'var(--teal)', 'var(--sea)', 'var(--green)'
];

async function loadManifest() {
  const response = await fetch(new URL('../manifest.json', import.meta.url));
  if (!response.ok) throw new Error('Simulation manifest could not be loaded.');
  return response.json();
}

async function loadMotion(filename, frames) {
  if (!motionBuffers.has(filename)) {
    motionBuffers.set(filename, fetch(new URL(`../motions/${filename}`, import.meta.url)).then(async (response) => {
      if (!response.ok) throw new Error(`Motion download failed (${response.status}).`);
      const data = new Float32Array(await response.arrayBuffer());
      if (data.length !== frames * 36 || !data.every(Number.isFinite)) {
        throw new Error('Motion data is incomplete.');
      }
      return data;
    }));
  }
  return motionBuffers.get(filename);
}

function createCard(scenario, index) {
  const clip = scenario.clips[0];
  const card = document.createElement('article');
  card.className = 'simulation-card';
  card.dataset.scenario = scenario.id;
  card.innerHTML = `
    <div class="simulation-stage" data-view="split" tabindex="0" aria-label="${scenario.showcase_title} synchronized simulation replay">
      <canvas aria-label="${scenario.showcase_title}: Viser and MuJoCo synchronized replay"></canvas>
      <div class="simulation-heading simulation-heading-left">
        <span class="simulation-heading-number">01 / GENERATED MOTION</span>
        <span class="simulation-heading-name">Viser</span>
        <span class="simulation-heading-description">Powered by Three.js</span>
      </div>
      <div class="simulation-heading simulation-heading-right">
        <span class="simulation-heading-number">02 / PHYSICS EXECUTION</span>
        <span class="simulation-heading-name">MuJoCo</span>
        <span class="simulation-heading-description">Driven by SONIC</span>
      </div>
      <div class="simulation-time"><span class="simulation-elapsed">0.00</span><span> / ${clip.duration.toFixed(2)} s</span></div>
      <button class="simulation-play" type="button" aria-label="Pause ${scenario.showcase_title}" aria-pressed="true">Ⅱ</button>
      <div class="simulation-loading" role="status"><span></span><small>Loading motion</small></div>
    </div>
    <div class="prompt-timeline simulation-timeline" role="slider" tabindex="0" aria-label="Prompt timeline for ${scenario.showcase_title}" aria-valuemin="0">
      <div class="prompt-track"></div>
    </div>`;

  const stage = card.querySelector('.simulation-stage');
  const canvas = card.querySelector('canvas');
  const playButton = card.querySelector('.simulation-play');
  const elapsed = card.querySelector('.simulation-elapsed');
  const loading = card.querySelector('.simulation-loading');
  const timeline = card.querySelector('.simulation-timeline');
  const track = card.querySelector('.prompt-track');

  const player = {
    index,
    scenario,
    clip,
    card,
    stage,
    canvas,
    playButton,
    elapsed,
    loading,
    timeline,
    track,
    viewer: null,
    ready: false,
    loadingPromise: null,
    visible: false,
    playing: !reducedMotion,
    time: 0,
    chips: [],
    dragging: false,
    resumeAfterDrag: false,
    dragAnchorX: 0,
    dragAnchorTranslate: 0,
    dragPlayheadX: 0
  };

  buildTimeline(player);
  bindPlayer(player);
  return player;
}

function buildTimeline(player) {
  const { clip, track } = player;
  player.chips = clip.segments.map((segment, index) => {
    const start = segment.start_frame / clip.fps;
    const end = segment.end_frame / clip.fps;
    const segmentDuration = Math.max(.8, end - start);
    const colorIndex = Math.round(index * (promptColors.length - 1) / Math.max(1, clip.segments.length - 1));
    const chip = document.createElement('div');
    chip.className = 'prompt-chip';
    chip.style.setProperty('--segment-color', promptColors[colorIndex]);
    chip.style.setProperty('--segment-accent', promptAccents[colorIndex]);
    chip.style.setProperty('--segment-grow', segmentDuration);
    chip.style.setProperty('--segment-width', `${Math.max(150, segmentDuration * 17, segment.prompt.length * 7.1 + 42)}px`);
    chip.dataset.start = start;
    chip.dataset.end = end;
    chip.title = segment.prompt;
    const label = document.createElement('span');
    label.textContent = segment.prompt;
    chip.append(label);
    track.append(chip);
    return chip;
  });
  player.timeline.setAttribute('aria-valuemax', clip.duration.toFixed(2));
  player.timeline.title = 'Drag left or right to scrub this motion';
}

function updateTimeline(player) {
  if (!player.chips.length) return;
  const time = player.time;
  let activeIndex = player.chips.findLastIndex((chip) => time >= Number(chip.dataset.start));
  if (activeIndex < 0) activeIndex = 0;

  player.chips.forEach((chip, index) => {
    const start = Number(chip.dataset.start);
    const end = Number(chip.dataset.end);
    const progress = Math.max(0, Math.min(1, (time - start) / Math.max(.01, end - start)));
    chip.classList.toggle('is-active', index === activeIndex);
    chip.classList.toggle('is-past', index < activeIndex);
    chip.style.setProperty('--segment-progress', `${progress * 100}%`);
  });

  const active = player.chips[activeIndex];
  const start = Number(active.dataset.start);
  const end = Number(active.dataset.end);
  const progress = Math.max(0, Math.min(1, (time - start) / Math.max(.01, end - start)));
  const playhead = active.offsetLeft + active.offsetWidth * progress;
  const desired = player.timeline.clientWidth * .34 - playhead;
  const minimum = Math.min(0, player.timeline.clientWidth - player.track.scrollWidth);

  if (!player.dragging) {
    player.track.style.transform = `translate3d(${Math.max(minimum, Math.min(0, desired))}px,0,0)`;
  }

  const matrix = new DOMMatrixReadOnly(getComputedStyle(player.track).transform);
  const visiblePlayhead = player.dragging
    ? player.dragPlayheadX
    : Math.max(0, Math.min(player.timeline.clientWidth, playhead + matrix.m41));
  player.timeline.style.setProperty('--playhead-position', `${visiblePlayhead}px`);
  player.timeline.setAttribute('aria-valuenow', time.toFixed(2));

}

function setPlayerTime(player, value, render = true) {
  player.time = Math.max(0, Math.min(player.clip.duration, value));
  player.elapsed.textContent = player.time.toFixed(2);
  updateTimeline(player);
  if (player.ready && render) {
    player.viewer.setTime(player.time);
    player.viewer.render();
  }
}

function setPlaying(player, playing) {
  player.playing = playing;
  player.playButton.textContent = playing ? 'Ⅱ' : '▶';
  player.playButton.setAttribute('aria-pressed', String(playing));
  player.playButton.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${player.scenario.showcase_title}`);
}

function timeAtTrackPosition(player, position) {
  const bounded = Math.max(0, Math.min(player.track.scrollWidth, position));
  const chip = player.chips.find((item) => bounded <= item.offsetLeft + item.offsetWidth) ?? player.chips.at(-1);
  const start = Number(chip.dataset.start);
  const end = Number(chip.dataset.end);
  const local = Math.max(0, Math.min(1, (bounded - chip.offsetLeft) / Math.max(1, chip.offsetWidth)));
  return start + local * (end - start);
}

function scrubFromDrag(player, event) {
  const minimum = player.dragPlayheadX - player.track.scrollWidth;
  const maximum = player.dragPlayheadX;
  const delta = event.clientX - player.dragAnchorX;
  const translate = Math.max(minimum, Math.min(maximum, player.dragAnchorTranslate + delta));
  player.track.style.transform = `translate3d(${translate}px,0,0)`;
  setPlayerTime(player, timeAtTrackPosition(player, player.dragPlayheadX - translate));
}

function bindPlayer(player) {
  player.playButton.addEventListener('click', async () => {
    await initializePlayer(player);
    if (player.time >= player.clip.duration - .01) setPlayerTime(player, 0);
    setPlaying(player, !player.playing);
  });

  player.stage.addEventListener('keydown', async (event) => {
    if (event.code !== 'Space') return;
    event.preventDefault();
    await initializePlayer(player);
    setPlaying(player, !player.playing);
  });

  player.timeline.addEventListener('pointerdown', async (event) => {
    if (event.button !== 0) return;
    await initializePlayer(player);
    player.dragAnchorX = event.clientX;
    player.dragAnchorTranslate = new DOMMatrixReadOnly(getComputedStyle(player.track).transform).m41;
    player.dragPlayheadX = parseFloat(getComputedStyle(player.timeline).getPropertyValue('--playhead-position')) || 0;
    player.dragging = true;
    player.resumeAfterDrag = player.playing;
    setPlaying(player, false);
    player.timeline.classList.add('is-dragging');
    player.timeline.setPointerCapture(event.pointerId);
    player.track.style.transition = 'none';
  });

  player.timeline.addEventListener('pointermove', (event) => {
    if (player.dragging) scrubFromDrag(player, event);
  });

  const finishDrag = (event) => {
    if (!player.dragging) return;
    scrubFromDrag(player, event);
    player.dragging = false;
    player.timeline.classList.remove('is-dragging');
    player.track.style.removeProperty('transition');
    if (player.timeline.hasPointerCapture(event.pointerId)) player.timeline.releasePointerCapture(event.pointerId);
    if (player.resumeAfterDrag) setPlaying(player, true);
  };

  player.timeline.addEventListener('pointerup', finishDrag);
  player.timeline.addEventListener('pointercancel', finishDrag);
  player.timeline.addEventListener('keydown', async (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    await initializePlayer(player);
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    setPlayerTime(player, player.time + direction * (event.shiftKey ? 5 : 1));
  });
}

async function initializePlayer(player) {
  if (player.ready) return;
  if (player.loadingPromise) return player.loadingPromise;
  player.loading.classList.add('is-visible');
  player.loadingPromise = (async () => {
    const viewer = new ReplayViewer(player.canvas, { mode: activeView });
    player.viewer = viewer;
    await viewer.loadRobot((message) => {
      player.loading.querySelector('small').textContent = message.replace('Preparing ', 'Loading ');
    });
    const [reference, actual] = await Promise.all([
      loadMotion(player.clip.file, player.clip.frames),
      loadMotion(player.clip.execution.file, player.clip.execution.frames)
    ]);
    viewer.setMotion(reference, player.clip, actual);
    viewer.setViewMode(activeView);
    player.stage.dataset.view = activeView;
    player.ready = true;
    player.loading.classList.remove('is-visible');
    setPlayerTime(player, player.time);
  })().catch((error) => {
    player.loading.classList.add('is-visible', 'is-error');
    player.loading.querySelector('small').textContent = error.message;
    console.error(error);
  });
  return player.loadingPromise;
}

function setView(mode) {
  activeView = mode;
  switchButtons.forEach((button) => {
    const selected = button.dataset.simulationView === mode;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  players.forEach((player) => {
    player.stage.dataset.view = mode;
    if (player.ready) {
      player.viewer.setViewMode(mode);
      player.viewer.render();
    }
  });
}

function animate(now) {
  const delta = Math.min((now - lastFrame) / 1000, .1);
  lastFrame = now;
  if (!document.hidden) {
    players.forEach((player) => {
      if (!player.ready || !player.visible) return;
      if (player.playing && !player.dragging) {
        const next = player.time + delta;
        setPlayerTime(player, next >= player.clip.duration ? next % player.clip.duration : next, false);
        player.viewer.setTime(player.time);
      }
      player.viewer.render();
    });
  }
  requestAnimationFrame(animate);
}

async function start() {
  try {
    const manifest = await loadManifest();
    manifest.scenarios.forEach((scenario, index) => {
      const player = createCard(scenario, index);
      players.push(player);
      grid.append(player.card);
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const player = players.find((item) => item.card === entry.target);
        player.visible = entry.isIntersecting;
        if (entry.isIntersecting) initializePlayer(player);
      });
    }, { rootMargin: '360px 0px', threshold: .01 });
    players.forEach((player) => observer.observe(player.card));
    switchButtons.forEach((button) => button.addEventListener('click', () => setView(button.dataset.simulationView)));
    setView('split');
    requestAnimationFrame(animate);
  } catch (error) {
    grid.innerHTML = `<p class="simulation-error">${error.message}</p>`;
    console.error(error);
  }
}

start();
