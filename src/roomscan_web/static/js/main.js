// Page state and wiring. The only module that touches both the API and the DOM.
import * as api from './api.js';
import {
  MAX_COMPARE, fillPresets, markColorMode, markFocus, needsReference, renderCells, renderJobs, renderLegend,
  renderMetrics, setCellNote, tickTimers,
} from './panels.js';
import { Viewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const state = {
  jobs: [],
  items: new Map(),        // key -> item, for everything that can be shown right now
  compare: [],             // keys ticked "compare" (max MAX_COMPARE), in the order they were ticked
  solo: null,              // key opened by clicking a row while nothing is ticked
  focus: null,             // key the metric panel talks about: clicked row or hovered cell
  zUp: new Map(),          // key -> Z-up ticked by hand; items not in here follow their own `up`
  colorMode: 'real',       // 'real' | 'error' (distance to the Faro reference)
  notes: new Map(),        // key -> { count, color }: what the small print under a cell says
  cellKeys: '',            // keys the cells were last built for
  fetchedAt: performance.now(),
};
const viewer = new Viewer($('view'));

// what the viewer shows: the ticked items, else the one opened row
function shownItems() {
  const keys = state.compare.length ? state.compare : state.solo ? [state.solo] : [];
  return keys.map((k) => state.items.get(k)).filter(Boolean);
}

const zUpOf = (item) => state.zUp.get(item.key) ?? item.up === 'z';

const COLOR_NOTE = { loading: 'computing error…', none: 'no Faro reference', failed: 'error colours failed' };

function showNote(index, key, part) {
  const note = { ...state.notes.get(key), ...part };
  state.notes.set(key, note);
  const count = note.count === undefined ? 'loading…' : note.count == null ? 'failed to load' : `${note.count.toLocaleString()} vertices`;
  setCellNote($('cells'), index, [count, COLOR_NOTE[note.color]].filter(Boolean).join(' · '));
}

function setColorMode(mode) {
  state.colorMode = mode;
  markColorMode($('color'), mode);
  $('legend').hidden = mode !== 'error';
  viewer.setColorMode(mode);
}

function setFocus(key) {
  state.focus = key;
  markFocus($('cells'), key);
  renderMetrics($('metrics'), state.items.get(key) ?? null);
}

function renderSidebar() {
  renderJobs($('jobs'), state.jobs, {
    shown: shownItems().map((it) => it.key),
    compare: state.compare,
    onPick: (key) => {
      if (!state.compare.includes(key)) { state.compare = []; state.solo = key; }
      state.focus = key;
      update();
    },
    onCompare: (key, on) => {
      if (on && state.compare.length >= MAX_COMPARE) return;
      state.compare = on ? [...state.compare, key] : state.compare.filter((k) => k !== key);
      state.solo = null;
      state.focus = on ? key : state.compare.at(-1) ?? null;
      update();
    },
  });
  tickTimers(state.fetchedAt);
}

function update() {
  renderSidebar();
  const items = shownItems();
  const keys = items.map((it) => it.key).join('|');
  if (keys !== state.cellKeys) {              // rebuild cells only when the selection changed: panes are live
    state.cellKeys = keys;
    const panes = renderCells($('cells'), items);
    $('zup').checked = items.length > 0 && items.every(zUpOf);
    items.forEach((item, i) => showNote(i, item.key, {}));
    viewer.setCells(items.map((item, i) => ({ item, pane: panes[i], zUp: zUpOf(item) })),
      (i, count) => showNote(i, items[i].key, { count }));
  }
  $('hud').hidden = items.length > 0;
  const hasReference = items.some((it) => it.urls.reference);
  needsReference($('overlay-label'), [$('overlay')], hasReference);
  needsReference($('color-label'), $('color').querySelectorAll('button'), hasReference);
  if (!hasReference) {                          // nothing to compare against: back to plain colours, overlay off
    $('overlay').checked = false;
    if (state.colorMode !== 'real') setColorMode('real');
  }
  viewer.setOverlay($('overlay').checked);
  const scenes = new Set(items.map((it) => it.scene).filter((s) => s != null));
  $('banner').hidden = scenes.size < 2;
  if (!items.some((it) => it.key === state.focus)) state.focus = items[0]?.key ?? null;
  setFocus(state.focus);
}

async function refresh() {
  state.jobs = await api.listScans();
  state.fetchedAt = performance.now();
  state.items = new Map(state.jobs.filter((j) => j.status === 'done').map((j) => [`web:${j.id}`, api.itemFromJob(j)]));
  state.compare = state.compare.filter((k) => state.items.has(k));
  update();
  if (state.jobs.some((j) => j.status === 'queued' || j.status === 'running')) setTimeout(refresh, 2000);
}

viewer.onColorState = (i, colorState) => {
  const item = shownItems()[i];
  if (item) showNote(i, item.key, { color: colorState });
};
viewer.onHover = (i) => { const item = shownItems()[i]; if (item && item.key !== state.focus) setFocus(item.key); };
$('overlay').onchange = () => viewer.setOverlay($('overlay').checked);
for (const b of $('color').querySelectorAll('button')) b.onclick = () => setColorMode(b.dataset.mode);
renderLegend($('legend'));
$('zup').onchange = () => {
  for (const item of shownItems()) state.zUp.set(item.key, $('zup').checked);
  viewer.setZUp($('zup').checked);
};
$('up').onsubmit = async (e) => {
  e.preventDefault();
  try {
    await api.createScan(new FormData(e.target));
  } catch (err) {
    alert(err.message);
    return;
  }
  e.target.reset(); $('preset').onchange(); refresh();
};
setInterval(() => tickTimers(state.fetchedAt), 500);
fillPresets($('preset'), $('preset-note'), await api.listPresets());
refresh();
