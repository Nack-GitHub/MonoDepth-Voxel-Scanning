// Pure formatters: no DOM, no state, so they are easy to eyeball and reuse.
export const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
export const eta = (s) => (s < 60 ? `${Math.ceil(s)} s` : mmss(s));
export const cm = (m) => `${(m * 100).toFixed(1)} cm`;
// use-case verdict from a recall value: passes, borderline, fails
export const badge = (recall) => (recall >= 0.9 ? '✓' : recall >= 0.75 ? '~' : '✗');
