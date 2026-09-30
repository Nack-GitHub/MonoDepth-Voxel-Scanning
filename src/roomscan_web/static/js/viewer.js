// three.js viewer: one renderer drawing 1-3 side-by-side cells, all through the same camera.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const loader = new PLYLoader();

// robust bounds: median vertex as centre, 90th-percentile radius as size (floaters would blow up a bbox)
function robustSphere(geo, matrixWorld) {
  const pos = geo.attributes.position, step = Math.max(1, Math.floor(pos.count / 20000));
  const xs = [], ys = [], zs = [], v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += step) { v.fromBufferAttribute(pos, i).applyMatrix4(matrixWorld); xs.push(v.x); ys.push(v.y); zs.push(v.z); }
  const med = (a) => { const b = a.slice().sort((p, q) => p - q); return b[b.length >> 1]; };
  const center = new THREE.Vector3(med(xs), med(ys), med(zs));
  const d = xs.map((x, i) => Math.hypot(x - center.x, ys[i] - center.y, zs[i] - center.z)).sort((p, q) => p - q);
  return { center, radius: d[Math.floor(d.length * 0.9)] };
}

// One viewport: its own scene (so meshes never overlap) but never its own camera.
class Cell {
  constructor(item, pane) {
    this.item = item;
    this.pane = pane;                 // DOM element whose rectangle this cell is drawn into
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 0.6); dir.position.set(2, 3, 4); this.scene.add(dir);
    this.group = new THREE.Group();   // carries the z-up -> y-up rotation for everything in the cell
    this.scene.add(this.group);
    this.mesh = null;
    this.disposed = false;
    // resolves to the vertex count, or null when the mesh could not be loaded
    this.ready = this.load().catch((err) => { console.error(err); return null; });
  }

  async load() {
    const geo = await loader.loadAsync(this.item.urls.mesh);
    if (this.disposed) { geo.dispose(); return null; }      // deselected while the file was on its way
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: geo.hasAttribute('color'), side: THREE.DoubleSide, flatShading: false });
    this.mesh = new THREE.Mesh(geo, mat);
    this.group.add(this.mesh);
    return geo.attributes.position.count;
  }

  setZUp(zUp) {
    this.group.rotation.x = zUp ? -Math.PI / 2 : 0;
    this.group.updateMatrixWorld(true);
  }

  dispose() {
    this.disposed = true;
    this.group.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
  }
}

export class Viewer {
  constructor(view) {
    this.view = view;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    view.prepend(this.renderer.domElement);       // under the cell captions
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.cells = [];
    this.generation = 0;
    this.onHover = null;                          // (cell index) => void
    this.bg = new THREE.Color();
    const scheme = matchMedia('(prefers-color-scheme: dark)');
    const readBg = () => this.bg.set(getComputedStyle(document.body).getPropertyValue('--bg').trim());
    scheme.addEventListener('change', readBg); readBg();
    new ResizeObserver(() => this.resize()).observe(view);
    this.renderer.domElement.addEventListener('pointermove', (e) => {
      const i = this.cells.findIndex((c) => { const b = c.pane.getBoundingClientRect(); return e.clientX >= b.left && e.clientX < b.right; });
      if (i >= 0 && this.onHover) this.onHover(i);
    });
    const loop = () => { this.controls.update(); this.render(); requestAnimationFrame(loop); };
    loop();
  }

  resize() {
    const w = this.view.clientWidth, h = this.view.clientHeight;
    // updateStyle must stay on: with it off the canvas takes its (pixelRatio x) buffer size as CSS size,
    // the grid track grows to fit it, and the framed mesh lands off-screen on Retina displays
    this.renderer.setPixelRatio(devicePixelRatio); this.renderer.setSize(w, h);
    this.render();      // setSize clears the buffer: redraw now, not a frame later, or a resize flashes blank
  }

  // Every cell is a scissored viewport of the one canvas. Rectangles are read from the DOM in CSS px
  // (the renderer applies the pixel ratio itself), so resizing and Retina need no extra bookkeeping.
  render() {
    const r = this.renderer, canvas = r.domElement.getBoundingClientRect();
    r.setScissorTest(false); r.setClearColor(this.bg); r.clear();
    r.setScissorTest(true);
    for (const cell of this.cells) {
      const b = cell.pane.getBoundingClientRect();
      if (b.width < 1 || b.height < 1) continue;
      const x = b.left - canvas.left, y = canvas.bottom - b.bottom;     // GL origin is bottom-left
      r.setViewport(x, y, b.width, b.height); r.setScissor(x, y, b.width, b.height);
      const aspect = b.width / b.height;
      if (this.camera.aspect !== aspect) { this.camera.aspect = aspect; this.camera.updateProjectionMatrix(); }
      r.render(cell.scene, this.camera);
    }
  }

  // Show `specs` = [{ item, pane, zUp }], one cell each, reusing cells whose item is already loaded.
  // `onLoaded(index, vertexCount | null)` reports each cell; the camera is framed once all are in.
  async setCells(specs, onLoaded) {
    const gen = ++this.generation;
    const old = new Map(this.cells.map((c) => [c.item.key, c]));
    this.cells = specs.map(({ item, pane, zUp }) => {
      let cell = old.get(item.key);
      if (cell) { old.delete(item.key); cell.pane = pane; cell.item = item; } else cell = new Cell(item, pane);
      cell.setZUp(zUp);
      return cell;
    });
    for (const cell of old.values()) cell.dispose();
    await Promise.all(this.cells.map(async (cell, i) => {
      const count = await cell.ready;
      if (gen === this.generation && onLoaded) onLoaded(i, count);
    }));
    if (gen === this.generation) this.frame();
  }

  setZUp(zUp) {
    for (const cell of this.cells) cell.setZUp(zUp);
    this.frame();
  }

  // One framing for every cell, taken from a single mesh (the LiDAR cell, else the first). Fitting each
  // cell to its own mesh would shrink an inflated room until it looked right — the error this page shows.
  frame() {
    const src = this.cells.find((c) => c.mesh && c.item.isLidar) ?? this.cells.find((c) => c.mesh);
    if (!src) return;
    src.group.updateMatrixWorld(true);
    const { center, radius } = robustSphere(src.mesh.geometry, src.mesh.matrixWorld);
    const { camera, controls } = this;
    // far enough that the sphere fits the narrower of the two view angles (cells are portrait in compare mode)
    const b = src.pane.getBoundingClientRect(), aspect = b.height > 0 ? b.width / b.height : 1;
    const half = Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.min(1, aspect));
    const dist = Math.max(2.4, 1.15 / Math.sin(half)) * radius;
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(0.7, 0.6, 0.8).normalize().multiplyScalar(dist));
    camera.near = radius / 100; camera.far = radius * 60; camera.updateProjectionMatrix();
  }
}
