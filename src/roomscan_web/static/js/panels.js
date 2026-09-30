// DOM for the sidebar: preset dropdown and the job list. No fetches, no three.js.
import { cm, eta, mmss } from './format.js';

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

// Newest first. `onPick(job)` fires for finished jobs only.
export function renderJobs(ul, jobs, active, onPick) {
  ul.replaceChildren();
  for (const j of jobs.slice().reverse()) {
    const li = el('li', j.id === active ? 'active' : '');
    const head = el('div', '', `${j.label} `);
    head.append(...statusSpans(j));
    li.append(head, el('div', 'st', j.id));
    const m = j.result?.metrics_3d;
    if (m) li.append(el('div', 'st', `chamfer ${cm(m.chamfer)} · F@5cm ${m['fscore@0.05'].toFixed(2)}`));
    else if (j.result) li.append(el('div', 'st', `${j.result.n_frames} frames · ${j.result.timing.total.toFixed(0)} s`));
    else if (j.error) li.append(el('div', 'st failed', j.error));
    li.onclick = () => { if (j.status === 'done') onPick(j); };
    ul.appendChild(li);
  }
}

// running timers count from the server's elapsed_s (+ time since that answer), so a reload does not reset them
export function tickTimers(fetchedAt) {
  const dt = (performance.now() - fetchedAt) / 1000;
  for (const e of document.querySelectorAll('[data-elapsed]')) e.textContent = `running ${mmss(+e.dataset.elapsed + dt)}`;
}
