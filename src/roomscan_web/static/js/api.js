// fetch wrappers: every request the page makes goes through here.
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export const listPresets = () => getJSON('/presets');
export const listScans = () => getJSON('/scans');

// FormData with `file` and `preset`; the server merges the preset's overrides.
export async function createScan(formData) {
  const r = await fetch('/scans', { method: 'POST', body: formData });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// An "item" is anything the viewer can show. Web jobs and gallery runs share this shape, so the
// viewer, the compare mode and the panels never need to know where a mesh came from.
export function itemFromJob(j) {
  return {
    key: `web:${j.id}`,
    origin: 'web',
    label: j.label,
    // a demo run is not a paper number: it was fused from a thinned capture and without the GT mask
    originNote: j.capture_stride ? `stride ${j.capture_stride}, no GT mask` : 'no GT mask',
    scene: j.scene ?? null,
    up: j.up ?? 'y',
    metrics: j.result ?? null,                          // the run's metrics.json
    isLidar: j.result?.depth_source === 'lidar',
    urls: { mesh: `/scans/${j.id}/mesh.ply` },
  };
}
