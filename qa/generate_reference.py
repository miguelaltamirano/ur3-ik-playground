"""Independent native-Pinocchio numerical references; no web/student dependency.
Run with an environment containing pinocchio and numpy. Uses the checked-in
numerical URDF, independently of the browser's compact transform implementation.
"""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
LENGTH = 0.4
model = pin.buildModelFromUrdf(str(ROOT / 'models/ur3_2f140.urdf'))
data = model.createData()
frame = model.getFrameId('classroom_tcp')
assert model.nq == model.nv == 6


def native(q):
    q = np.asarray(q)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p, R = data.oMf[frame].translation.copy(), data.oMf[frame].rotation.copy()
    J = pin.computeFrameJacobian(model, data, q, frame, pin.LOCAL_WORLD_ALIGNED).copy()
    return p, R, J


def sample(name, q):
    p, R, J = native(q)
    scaled = np.vstack([J[:3] / LENGTH, J[3:]])
    return dict(name=name, q=np.asarray(q).tolist(), p=p.tolist(), R=R.tolist(),
                J=J.tolist(), svdPosition=np.linalg.svd(J[:3]/LENGTH, compute_uv=False).tolist(),
                svdPose=np.linalg.svd(scaled, compute_uv=False).tolist())


def main():
    poses = {
        'comfort': np.deg2rad([0, -90, 90, -90, 90, 0]),
        'wrist-near': np.deg2rad([0, -90, 90, -90, 0.5, 0]),
        'wrist-exact': np.deg2rad([0, -90, 90, -90, 0, 0]),
        'elbow-exact': np.deg2rad([0, -65, 0, -40, -70, 15]),
    }
    rng = np.random.default_rng(1403)
    samples = [sample(name, q) for name, q in poses.items()]
    samples += [sample(f'random-{i:02}', rng.uniform(-2.5, 2.5, 6)) for i in range(16)]
    spec = importlib.util.spec_from_file_location('kernel', ROOT / 'dist/python/kernel.py')
    kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)
    compact_model = json.loads((ROOT/'dist/assets/robot/kinematics.json').read_text())
    fk = kernel.make_kinematics(compact_model)
    compact_errors = []
    for reference in samples:
        p, R, J = fk(reference['q'])
        compact_errors.append(max(np.max(np.abs(p-reference['p'])), np.max(np.abs(R-reference['R'])), np.max(np.abs(J-reference['J']))))
    # Central differences test the negative residual derivative, not the geometric
    # Jacobian; random independent target rotations exercise the SO(3) correction.
    residual_errors = []
    for i in range(25):
        q = rng.uniform(-2, 2, 6)
        p, R, _ = native(q)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.05, 2.8)
        target = {'p': p + rng.normal(size=3)*0.05, 'R': pin.exp3(axis*angle) @ R}
        task = kernel.make_task(fk, 'pose'); _, J = task(q, target)
        h = 1e-6; finite = np.empty((6, 6))
        for k in range(6):
            dq = np.zeros(6); dq[k] = h
            finite[:, k] = -(task(q+dq, target)[0]-task(q-dq, target)[0]) / (2*h)
        residual_errors.append(float(np.max(np.abs(J-finite))))
    source = (ROOT/'dist/python/dls.py').read_text()
    perturbation = np.deg2rad([4, -4, 5, 3, -4, 3])
    trials = []
    for name, q in poses.items():
        p, R, _ = native(q+perturbation)
        target = {'p': p.tolist(), 'R': R.tolist()}
        for mode in ('pose', 'position'):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                result = kernel.run_student(source, {'start': q.tolist(), 'target': target, 'mode':mode, 'damping':0.03}, compact_model)
            trials.append(dict(name=name, mode=mode, damping=.03, targetQ=(q+perturbation).tolist(), target=target,
                               converged=result['converged'], updates=result['updates'], positionError=result['error'],
                               orientationError=result['angle'], finalQ=result['trajectory'][-1], output=stream.getvalue()))
    report = dict(generator='Native Pinocchio '+pin.__version__, frame='world', tcp='classroom_tcp',
                  jacobianRows='linear xyz followed by angular xyz; LOCAL_WORLD_ALIGNED', characteristicLength=LENGTH,
                  samples=samples, solverTrials=trials,
                  validation=dict(compactKinematicsMaxAbsError=max(compact_errors),
                                  effectiveTaskJacobianMaxAbsFiniteDifferenceError=max(residual_errors),
                                  orientationDerivativeSamples=25, centralDifferenceStep=1e-6,
                                  orientationTargetAngleRangeRadians=[0.05,2.8]))
    assert max(compact_errors) < 1e-12
    assert max(residual_errors) < 1e-7
    assert all(t['converged'] for t in trials if t['name']=='comfort')
    (ROOT/'qa/reference.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'validation':report['validation'], 'poses':[{k:s[k] for k in ('name','p','svdPosition','svdPose')} for s in samples[:4]],
                      'solverTrials':[{k:t[k] for k in ('name','mode','converged','updates','positionError','orientationError')} for t in trials]}, indent=2))

if __name__=='__main__': main()
