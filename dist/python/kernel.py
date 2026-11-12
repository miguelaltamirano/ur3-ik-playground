"""UR3 + fixed-open Robotiq 2F-85: NumPy kinematics and IK harness.
Geometric Jacobians are expressed in the model world/base frame.
Full-pose diagnostics use [J_position / 0.4 m; J_angular].
"""
import json
import time
import numpy as np

LENGTH = 0.4
POSITION_TOL = 1e-4
ORIENTATION_TOL = np.deg2rad(0.1)


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rotation_log(R):
    cosine = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cosine)
    vector = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    if angle < 1e-7:
        return vector * 0.5
    if np.pi - angle < 1e-5:
        i = int(np.argmax(np.diag(R)))
        axis = np.zeros(3)
        axis[i] = np.sqrt(max(0, (R[i, i]+1) / 2))
        for j in range(3):
            if i != j:
                axis[j] = (R[i, j]+R[j, i]) / (4*max(axis[i], 1e-12))
        if axis @ vector < 0:
            axis = -axis
        return axis * angle / max(np.linalg.norm(axis), 1e-12)
    return vector * angle / (2*np.sin(angle))


def right_log_jacobian_inverse(vector):
    theta = np.linalg.norm(vector)
    K = skew(vector)
    coefficient = 1/12 + theta**2/720 if theta < 1e-5 else 1/theta**2 - np.cos(theta/2)/(2*theta*np.sin(theta/2))
    return np.eye(3) + 0.5*K + coefficient*(K @ K)


def make_kinematics(model):
    def kinematics(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('kinematics(q) needs six finite joint angles.')
        p, R = np.zeros(3), np.eye(3)
        origins, axes = [], []
        for joint, angle in zip(model['joints'], q):
            p = p + R @ np.asarray(joint['xyz'])
            R = R @ np.asarray(joint['rotation'])
            axis = np.asarray(joint['axis'])
            origins.append(p.copy()); axes.append(R @ axis)
            K = skew(axis)
            R = R @ (np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K))
        p = p + R @ np.asarray(model['tool']['xyz'])
        R = R @ np.asarray(model['tool']['rotation'])
        Jv = np.column_stack([np.cross(axis, p-origin) for axis, origin in zip(axes, origins)])
        return p, R, np.vstack([Jv, np.column_stack(axes)])
    return kinematics


def make_task(kinematics, mode):
    def task(q, target):
        p, R, J = kinematics(q)
        error = (np.asarray(target['p']) - p) / LENGTH
        if mode == 'position':
            return error, J[:3] / LENGTH
        rotation_error = rotation_log(np.asarray(target['R']) @ R.T)
        # The correction is the derivative of the rotation-vector residual.
        # Singularity diagnostics separately use the geometric J, independent of target.
        return np.r_[error, rotation_error], np.vstack([J[:3]/LENGTH, right_log_jacobian_inverse(rotation_error) @ J[3:]])
    return task


def run_student(source, config, model):
    if len(source) > 50000:
        raise ValueError('Keep the exercise below 50,000 characters.')
    lower = np.array([j['lower'] for j in model['joints']])
    upper = np.array([j['upper'] for j in model['joints']])
    mode = config.get('mode', 'pose')
    if mode not in ('position', 'pose'):
        raise ValueError('Choose position or pose mode.')
    target = {'p': np.asarray(config['target']['p'], dtype=float), 'R': np.asarray(config['target']['R'], dtype=float)}
    if target['p'].shape != (3,) or target['R'].shape != (3, 3) or not np.isfinite(target['p']).all() or not np.isfinite(target['R']).all():
        raise ValueError('The target must have a finite position and 3x3 orientation.')
    if not np.allclose(target['R'].T @ target['R'], np.eye(3), atol=1e-6) or abs(np.linalg.det(target['R'])-1) > 1e-6:
        raise ValueError('Target orientation must be a rotation matrix.')
    kinematics = make_kinematics(model)
    task = make_task(kinematics, mode)
    trajectory = []

    def validate(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Return and record six finite joint angles, shape (6,).')
        if np.any(q < lower-1e-9) or np.any(q > upper+1e-9):
            raise ValueError('A joint exceeded its classroom limit. Apply np.clip(q, lower, upper).')
        return q.copy()

    def record(q):
        q = validate(q)
        if len(trajectory) >= 1001:
            raise ValueError('At most 1,000 recorded updates are allowed.')
        trajectory.append(q)

    def converged(q, target):
        p, R, _ = kinematics(q)
        return bool(np.linalg.norm(target['p']-p) < POSITION_TOL and
                    (mode == 'position' or np.linalg.norm(rotation_log(target['R'] @ R.T)) < ORIENTATION_TOL))

    q0 = validate(config['start']); record(q0)
    namespace = {'np': np, 'kinematics': kinematics, 'task': task, 'converged': converged,
                 'record': record, 'lower': lower.copy(), 'upper': upper.copy(),
                 'DAMPING': float(config.get('damping', 0.03)), '__name__': 'student_solver'}
    begin = time.perf_counter()
    exec(compile(source, 'student_solver.py', 'exec'), namespace)
    if not callable(namespace.get('solve_ik')):
        raise ValueError('Define solve_ik(q, target) and return the final q.')
    final_q = validate(namespace['solve_ik'](q0.copy(), {k: v.copy() for k, v in target.items()}))
    if not np.array_equal(final_q, trajectory[-1]): record(final_q)
    elapsed_ms = (time.perf_counter()-begin)*1000
    errors, angles, sigmas = [], [], []
    for q in trajectory:
        p, R, J = kinematics(q)
        errors.append(float(np.linalg.norm(target['p']-p)))
        angles.append(float(np.linalg.norm(rotation_log(target['R'] @ R.T))))
        scaled = J[:3]/LENGTH if mode == 'position' else np.vstack([J[:3]/LENGTH, J[3:]])
        sigmas.append(float(np.linalg.svd(scaled, compute_uv=False)[-1]))
    return {'trajectory': [q.tolist() for q in trajectory], 'errors': errors, 'angles': angles,
            'sigmas': sigmas, 'updates': len(trajectory)-1, 'error': errors[-1], 'angle': angles[-1],
            'converged': converged(final_q, target), 'elapsed_ms': elapsed_ms, 'mode': mode}
