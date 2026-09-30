// three.js viewer: one renderer, one mesh at a time.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

export class Viewer {
  constructor(view) {
    this.view = view;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    view.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 0.6); dir.position.set(2, 3, 4); this.scene.add(dir);
    this.mesh = null;
    addEventListener('resize', () => this.resize()); this.resize();
    this.scene.background = new THREE.Color(getComputedStyle(document.body).getPropertyValue('--bg').trim());
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); };
    loop();
  }

  resize() {
    const w = this.view.clientWidth, h = this.view.clientHeight;
    // updateStyle must stay on: with it off the canvas takes its (pixelRatio x) buffer size as CSS size,
    // the grid track grows to fit it, and the framed mesh lands off-screen on Retina displays
    this.renderer.setPixelRatio(devicePixelRatio); this.renderer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  // Load a PLY, replace the current mesh and frame the camera on it. Returns the vertex count.
  async show(url, zUp) {
    const geo = await new PLYLoader().loadAsync(url);
    geo.computeVertexNormals();
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
    const mat = new THREE.MeshStandardMaterial({ vertexColors: geo.hasAttribute('color'), side: THREE.DoubleSide, flatShading: false });
    const mesh = this.mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = zUp ? -Math.PI / 2 : 0;   // z-up capture -> y-up viewer
    this.scene.add(mesh);
    mesh.updateMatrixWorld(true);
    // robust framing: median vertex as centre, 90th-percentile radius as size (floaters would blow up a bbox)
    const pos = geo.attributes.position, step = Math.max(1, Math.floor(pos.count / 20000));
    const xs = [], ys = [], zs = [], v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i += step) { v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld); xs.push(v.x); ys.push(v.y); zs.push(v.z); }
    const med = (a) => { const b = a.slice().sort((p, q) => p - q); return b[b.length >> 1]; };
    const c = new THREE.Vector3(med(xs), med(ys), med(zs));
    const d = xs.map((x, i) => Math.hypot(x - c.x, ys[i] - c.y, zs[i] - c.z)).sort((p, q) => p - q);
    const s = 2 * d[Math.floor(d.length * 0.9)];
    const { camera, controls } = this;
    controls.target.copy(c); camera.position.copy(c).add(new THREE.Vector3(s * 0.7, s * 0.6, s * 0.8));
    camera.near = s / 200; camera.far = s * 20; camera.updateProjectionMatrix();
    return pos.count;
  }
}
