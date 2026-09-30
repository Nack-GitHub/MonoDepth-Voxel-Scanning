// Page state and wiring. The only module that touches both the API and the DOM.
import * as api from './api.js';
import { fillPresets, renderJobs, tickTimers } from './panels.js';
import { Viewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const state = { active: null, fetchedAt: performance.now() };
const viewer = new Viewer($('view'));

async function show(id) {
  $('hud').textContent = 'loading mesh…';
  const n = await viewer.show(api.meshUrl(id), $('zup').checked);
  $('hud').textContent = `${id}: ${n.toLocaleString()} vertices`;
}

async function refresh() {
  const jobs = await api.listScans();
  state.fetchedAt = performance.now();
  renderJobs($('jobs'), jobs, state.active, (j) => { state.active = j.id; show(j.id); refresh(); });
  tickTimers(state.fetchedAt);
  if (jobs.some((j) => j.status === 'queued' || j.status === 'running')) setTimeout(refresh, 2000);
}

$('zup').onchange = () => { if (state.active) show(state.active); };
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
