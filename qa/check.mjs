import fs from 'node:fs';
import assert from 'node:assert/strict';
import {makeKinematics,diagnostics,rpyMatrix,matrixRpy,rotationLog,rotation,norm,sub,mm,transpose} from '../dist/math.js';
import {createRobot} from '../dist/robot-model.js';
import * as THREE from '../dist/vendor/three.module.min.js';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url)));
const model=read('../dist/assets/robot/kinematics.json'),reference=read('./reference.json');
const forward=makeKinematics(model),robot=createRobot(read('../dist/assets/robot/robot.json'),model);
function near(a,b,tol=1e-9){const x=a.flat(Infinity),y=b.flat(Infinity);assert.equal(x.length,y.length);assert.ok(Math.max(...x.map((v,i)=>Math.abs(v-y[i])))<tol);}
for(const sample of reference.samples){
 const f=forward(sample.q);near(f.p,sample.p);near(f.R,sample.R);near(f.J,sample.J);
 near(diagnostics(f,'position').s,sample.svdPosition);near(diagnostics(f,'pose').s,sample.svdPose);
 robot.setPose(sample.q);near(robot.tool.getWorldPosition(new THREE.Vector3()).toArray(),f.p);
 const r=new THREE.Matrix4().extractRotation(robot.tool.matrixWorld).elements;
 near([[r[0],r[4],r[8]],[r[1],r[5],r[9]],[r[2],r[6],r[10]]],f.R);
 near(rpyMatrix(matrixRpy(f.R)),f.R,1e-7);
 // Finite-difference position and angular geometric Jacobians, independent of analytic cross products.
 for(let j=0;j<6;j++){const a=[...sample.q],b=[...sample.q],h=1e-6;a[j]+=h;b[j]-=h;const fa=forward(a),fb=forward(b);
  near(sub(fa.p,fb.p).map(v=>v/(2*h)),f.Jv.map(row=>row[j]),1e-8);
  near(rotationLog(mm(fa.R,transpose(fb.R))).map(v=>v/(2*h)),f.Jw.map(row=>row[j]),1e-8);
 }
}
for(const index of [2,3]){const f=forward(reference.samples[index].q);assert.equal(diagnostics(f).rank,5);assert.equal(diagnostics(f,'position').rank,3);}
for(const angle of [0,1e-9,.4,Math.PI-1e-7,Math.PI]){const axis=[1/3,2/3,2/3],R=rotation(axis,angle);near(rotation(axis,norm(rotationLog(R))),R,1e-7);}
console.log('PASS: 20 Pinocchio FK, world Jacobian and NumPy SVD references; renderer TCP alignment; finite differences; exact singularities; SO(3) edge cases.');
