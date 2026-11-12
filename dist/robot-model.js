import * as THREE from './vendor/three.module.min.js';


// URDF uses fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll).
const orientation = rpy => new THREE.Quaternion().setFromEuler(new THREE.Euler(...rpy, 'ZYX'));

export function createRobot(model, arm, buffer = null) {
  const root = new THREE.Group(), links = {}, joints = {}, geometries = {};
  root.name = 'UR3 + Robotiq 2F-85';
  if (buffer) for (const [name, m] of Object.entries(model.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buffer, m.vertexOffset, m.vertexCount * 3), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, m.indexOffset, m.indexCount), 1));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    geometries[name] = g;
  }
  const armLinks = new Set();
  for (const l of model.links) {
    const link = links[l.name] = new THREE.Group();
    link.name = l.name;
    if (!buffer) continue;
    for (const v of l.visuals) {
      const color = new THREE.Color().setRGB(...v.rgba.slice(0, 3), THREE.SRGBColorSpace);
      if (armLinks.has(l.name)) color.lerp(new THREE.Color('#72b5a0'), .30);
      const material = new THREE.MeshStandardMaterial({color, metalness: .28, roughness: .48});
      const visual = new THREE.Mesh(geometries[v.mesh], material);
      visual.position.fromArray(v.xyz);
      visual.quaternion.copy(orientation(v.rpy));
      visual.scale.fromArray(v.scale);
      visual.castShadow = true;
      visual.receiveShadow = true;
      link.add(visual);
    }
  }
  for (const j of model.joints) {
    const origin = new THREE.Group();
    origin.position.fromArray(j.xyz);
    origin.quaternion.copy(orientation(j.rpy));
    const pivot = new THREE.Group();
    pivot.name = j.name;
    origin.add(pivot);
    pivot.add(links[j.child]);
    links[j.parent].add(origin);
    joints[j.name] = {pivot, axis: new THREE.Vector3(...j.axis).normalize()};
  }
  const children = new Set(model.joints.map(j => j.child));
  for (const l of model.links) if (!children.has(l.name)) root.add(links[l.name]);
  const tool = links.classroom_tcp;
  function setPose(q) {
    arm.joints.forEach((j, i) => joints[j.name].pivot.quaternion.setFromAxisAngle(joints[j.name].axis, q[i]));
    root.updateMatrixWorld(true);
  }
  return {root, links, joints, tool, setPose};
}
