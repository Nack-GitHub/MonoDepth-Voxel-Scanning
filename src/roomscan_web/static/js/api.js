// fetch wrappers: every request the page makes goes through here.
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export const listPresets = () => getJSON('/presets');
export const listScans = () => getJSON('/scans');
export const meshUrl = (id) => `/scans/${id}/mesh.ply`;

// FormData with `file` and `preset`; the server merges the preset's overrides.
export async function createScan(formData) {
  const r = await fetch('/scans', { method: 'POST', body: formData });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
