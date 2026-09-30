// DOM for everything around the canvas: preset dropdown, job list, cell captions. No fetches, no three.js.
import { badge, cm, eta, mmss } from './format.js';

export const MAX_COMPARE = 3;
const ORIGIN = { web: 'demo run', paper: 'paper run' };      // where a number comes from is always on screen
// Use cases of the paper and the recall each one is judged by. The thresholds are the paper's illustration
// of what a number means, not an industry standard — the panel says so.
const USE_CASES = [['Overview', 'recall@0.1', 10], ['Furniture', 'recall@0.05', 5], ['Renovation', 'recall@0.02', 2]];
const VERDICT = { '✓': 'ok', '~': 'mid', '✗': 'no' };
// errorcolor.TURBO sampled every 10 % (tests/test_web.py keeps the two in step)
const TURBO_STOPS = [
  '#30123b', '#455ccf', '#3e9bfe', '#19d5cd', '#46f884', '#a4fc3c', '#e1dd37', '#fea732', '#f05b12', '#c32503', '#7a0403',
];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// presets live on the server (GET /presets) so the labels match the paper in one place
export function fillPresets(select, note, presets) {
  for (const p of presets) select.add(new Option(p.label, p.key));
  select.onchange = () => { note.textContent = presets.find((p) => p.key === select.value)?.note ?? ''; };
  select.onchange();
}

function statusSpans(j) {
  if (j.status === 'running') {
    const run = el('span', 'st running');
    run.dataset.elapsed = j.elapsed_s ?? 0;
    return j.eta_s ? [run, ' ', el('span', 'st', `expected ~${eta(j.eta_s)}`)] : [run];
  }
  const took = j.status === 'done' && j.elapsed_s != null ? ` in ${mmss(j.elapsed_s)}` : '';
  return [el('span', `st ${j.status}`, j.status + took)];
}

// "compare" checkbox of a row; at most MAX_COMPARE can be ticked at once.
function compareBox(key, compare, onCompare) {
  const box = el('label', 'pick st', ' compare');
  const input = el('input');
  input.type = 'checkbox';
  input.checked = compare.includes(key);
  input.disabled = !input.checked && compare.length >= MAX_COMPARE;
  if (input.disabled) box.title = `up to ${MAX_COMPARE} at once`;
  input.onchange = () => onCompare(key, input.checked);
  box.onclick = (e) => e.stopPropagation();          // ticking must not also open the row
  box.prepend(input);
  return box;
}

// Newest first. Finished jobs can be opened (`onPick(key)`) or ticked for comparison (`onCompare(key, on)`).
export function renderJobs(ul, jobs, { shown, compare, onPick, onCompare }) {
  ul.replaceChildren();
  for (const j of jobs.slice().reverse()) {
    const key = `web:${j.id}`;
    const li = el('li', shown.includes(key) ? 'active' : '');
    const head = el('div', 'row');
    head.append(el('span', 'name', j.label));
    if (j.status === 'done') head.append(compareBox(key, compare, onCompare));
    const sub = el('div', 'st');
    sub.append(...statusSpans(j), ` · ${j.id}`);
    li.append(head, sub);
    const m = j.result?.metrics_3d;
    if (m) li.append(el('div', 'st', `chamfer ${cm(m.chamfer)} · F@5cm ${m['fscore@0.05'].toFixed(2)}`));
    else if (j.result) li.append(el('div', 'st', `${j.result.n_frames} frames · ${j.result.timing.total.toFixed(0)} s`));
    else if (j.error) li.append(el('div', 'st failed', j.error));
    li.onclick = () => { if (j.status === 'done') onPick(key); };
    ul.appendChild(li);
  }
}

// The three sources the demo compares; a click on a room's header ticks them in this order.
const DEMO_TRIO = ['mono_metric', 'mono_ft_lidar_all', 'lidar'];

// Gallery tab: sources down, rooms across, Chamfer (cm) in each cell. A cell is a "compare" toggle that
// counts towards the same MAX_COMPARE as the web jobs; `onRoom(keys)` compares the demo trio of one room.
export function renderGallery(box, runs, { compare, onCompare, onRoom }) {
  box.replaceChildren();
  if (!runs.length) {
    box.append(el('p', 'g-note', 'No paper runs found: the gallery lists experiment runs that still have their mesh.ply (meshes are not committed; they exist on the machine that ran the sweeps).'));
    return;
  }
  const scenes = [...new Set(runs.map((r) => r.scene))];
  const names = [...new Map(runs.map((r) => [r.run, r.label]))];          // [run, label], server order
  const at = new Map(runs.map((r) => [`${r.scene}/${r.run}`, r]));
  const full = compare.length >= MAX_COMPARE;
  const table = el('table');
  const head = el('tr');
  head.append(el('th', 'src', 'Chamfer, cm'));
  for (const scene of scenes) {
    const room = el('button', 'room', scene);
    room.type = 'button';
    room.title = 'compare pretrained / FT-LiDAR-24 / iPad LiDAR of this room';
    room.onclick = () => onRoom(DEMO_TRIO.map((run) => at.get(`${scene}/${run}`)).filter(Boolean).map((r) => `paper:${r.id}`));
    const th = el('th');
    th.append(room);
    head.append(th);
  }
  table.append(el('thead'), el('tbody'));
  table.tHead.append(head);
  for (const [run, label] of names) {
    const tr = el('tr');
    tr.append(el('th', 'src', label));
    for (const scene of scenes) {
      const r = at.get(`${scene}/${run}`), td = el('td');
      const chamfer = r?.metrics?.metrics_3d?.chamfer;
      if (!r) td.textContent = '—';
      else {
        const key = `paper:${r.id}`, on = compare.includes(key);
        const b = el('button', on ? 'on' : '', chamfer != null ? (chamfer * 100).toFixed(1) : '·');
        b.type = 'button';
        b.setAttribute('role', 'checkbox');
        b.setAttribute('aria-checked', String(on));
        b.disabled = !on && full;
        b.title = b.disabled ? `up to ${MAX_COMPARE} at once` : `${label} — room ${scene}`;
        b.onclick = () => onCompare(key, !on);
        td.append(b);
      }
      tr.append(td);
    }
    table.tBodies[0].append(tr);
  }
  box.append(
    el('p', 'g-note', 'Runs of the paper’s experiments. Click a cell to show it (up to 3 side by side), a room number for pretrained / FT-LiDAR-24 / LiDAR.'),
    table,
  );
}

// running timers count from the server's elapsed_s (+ time since that answer), so a reload does not reset them
export function tickTimers(fetchedAt) {
  const dt = (performance.now() - fetchedAt) / 1000;
  for (const e of document.querySelectorAll('[data-elapsed]')) e.textContent = `running ${mmss(+e.dataset.elapsed + dt)}`;
}

// One cell per shown item: a transparent pane the viewer draws into, and a caption under it.
// Returns the panes, in item order.
export function renderCells(container, items) {
  container.replaceChildren();
  container.style.gridTemplateColumns = `repeat(${Math.max(1, items.length)}, 1fr)`;
  return items.map((item) => {
    const cell = el('div', 'cell');
    cell.dataset.key = item.key;
    const pane = el('div', 'pane');
    const cap = el('div', 'cap');
    const chamfer = item.metrics?.metrics_3d?.chamfer;
    cap.append(
      el('div', 'cap-label', item.label),
      el('div', 'cap-num', chamfer != null ? `Chamfer ${cm(chamfer)}` : 'no 3D metrics'),
      el('div', 'st', ''),
    );
    cap.lastChild.append(originBadge(item), el('span', 'cap-note', ' loading…'));
    cell.append(pane, cap);
    container.appendChild(cell);
    return pane;
  });
}

function originBadge(item) {
  return el('span', `origin ${item.origin}`, ORIGIN[item.origin] + (item.originNote ? ` (${item.originNote})` : ''));
}

function tile(key, value) {
  const t = el('div', 'tile');
  t.append(el('div', 'k', key), el('div', 'v', value));
  return t;
}

// Every number of the focused item (`null` hides the panel): geometry metrics in cm, F-score and recalls,
// stage timings, and the three use-case verdicts.
export function renderMetrics(box, item) {
  box.replaceChildren();
  box.hidden = !item;
  if (!item) return;
  const head = el('div', 'm-head');
  head.append(el('span', 'm-title', item.label), originBadge(item));
  const m = item.metrics?.metrics_3d, t = item.metrics?.timing;
  const time = t ? `${t.depth.toFixed(0)} / ${t.fusion.toFixed(1)} / ${t.total.toFixed(0)} s` : '—';
  if (!m) {
    head.append(el('span', 'st', 'no Faro reference — no 3D metrics'));
    box.append(head, tile('Time: depth / fusion / total', time));
    return;
  }
  const badges = el('div', 'badges');
  for (const [name, key, cmAt] of USE_CASES) {
    const verdict = badge(m[key]);
    const chip = el('span', `uc ${VERDICT[verdict]}`, `${name} ${verdict}`);
    chip.title = `recall@${cmAt} cm = ${m[key].toFixed(2)} (✓ ≥ 0.90, ~ 0.75–0.90, ✗ < 0.75)`;
    badges.append(chip);
  }
  badges.append(el('div', 'm-note', 'thresholds are illustrative, not a standard'));
  head.append(badges);
  const tiles = el('div', 'tiles');
  tiles.append(
    tile('Chamfer', cm(m.chamfer)), tile('Accuracy', cm(m.accuracy)), tile('Completeness', cm(m.completeness)),
    tile('F@5 cm', m['fscore@0.05'].toFixed(2)), tile('Recall@2 cm', m['recall@0.02'].toFixed(2)),
    tile('Recall@5 cm', m['recall@0.05'].toFixed(2)), tile('Recall@10 cm', m['recall@0.1'].toFixed(2)),
    tile('Time: depth / fusion / total', time),
  );
  box.append(head, tiles);
}

// A toolbar control that needs the Faro reference: greyed out, with the reason as tooltip, when no shown item has one.
export function needsReference(label, inputs, available) {
  for (const input of inputs) input.disabled = !available;
  label.classList.toggle('off', !available);
  label.title = available ? '' : 'no Faro reference';
}

// The colour scale of the error mode: turbo, 0 to 10 cm, everything beyond in the last colour.
export function renderLegend(box) {
  const bar = el('div', 'bar');
  bar.style.background = `linear-gradient(to right, ${TURBO_STOPS.join(', ')})`;
  box.replaceChildren(el('span', '', 'error vs Faro'), el('span', '', '0'), bar, el('span', '', '≥ 10 cm'));
}

export function markColorMode(seg, mode) {
  for (const b of seg.querySelectorAll('button')) b.classList.toggle('on', b.dataset.mode === mode);
}

export function setCellNote(container, index, text) {
  const note = container.children[index]?.querySelector('.cap-note');
  if (note) note.textContent = ` ${text}`;
}

export function markFocus(container, key) {
  for (const cell of container.children) cell.classList.toggle('focus', cell.dataset.key === key);
}
