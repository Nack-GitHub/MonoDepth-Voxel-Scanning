// DOM for everything around the canvas: preset dropdown, job list, cell captions. No fetches, no three.js.
import { cm, eta, mmss } from './format.js';

export const MAX_COMPARE = 3;
const ORIGIN = { web: 'demo run', paper: 'paper run' };      // where a number comes from is always on screen

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
    const origin = ORIGIN[item.origin] + (item.originNote ? ` (${item.originNote})` : '');
    cap.lastChild.append(el('span', `origin ${item.origin}`, origin), el('span', 'cap-note', ' loading…'));
    cell.append(pane, cap);
    container.appendChild(cell);
    return pane;
  });
}

export function setCellNote(container, index, text) {
  const note = container.children[index]?.querySelector('.cap-note');
  if (note) note.textContent = ` ${text}`;
}

export function markFocus(container, key) {
  for (const cell of container.children) cell.classList.toggle('focus', cell.dataset.key === key);
}
