// Page state and wiring. The only module that touches both the API and the DOM.
import * as api from './api.js';
import {
  MAX_COMPARE, fillPresets, markFocus, needsReference, renderCells, renderJobs, setCellNote, tickTimers,
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

function setFocus(key) {
  state.focus = key;
  markFocus($('cells'), key);
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
    viewer.setCells(items.map((item, i) => ({ item, pane: panes[i], zUp: zUpOf(item) })), (i, count) => {
      setCellNote($('cells'), i, count == null ? 'failed to load' : `${count.toLocaleString()} vertices`);
    });
  }
  $('hud').hidden = items.length > 0;
  needsReference($('overlay-label'), $('overlay'), items.some((it) => it.urls.reference));
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

viewer.onHover = (i) => { const item = shownItems()[i]; if (item && item.key !== state.focus) setFocus(item.key); };
$('overlay').onchange = () => viewer.setOverlay($('overlay').checked);
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
