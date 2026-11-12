# UR3 + Robotiq 2F-85 inverse-kinematics lab

**Open the lab:** https://miguelaltamirano.github.io/ur3-ik-playground/

A separate classroom website for students who already know forward kinematics and Jacobians. Open the link in a modern browser; no installation or sign-in. Python and NumPy load from a pinned Pyodide CDN on the first run, so an internet connection is required. WebGL 2 provides the 3D view. The numerical editor and plots remain usable if WebGL is unavailable.

Students can move six joints, set a position or full-pose target, edit real Python, and compare damped least squares, the pseudoinverse and Jacobian transpose. Results include measured position/orientation residuals, playback, singular values, a wrist-alignment sweep, a normalized position-velocity ellipsoid and a five-run history. Draft code stays in local browser storage. Download exports a standalone Python file containing the current solver, kinematic model, task, start and target; running that file locally requires Python and NumPy.

## A short classroom sequence

1. Use **Comfortable pose → Run code**. Replay the three default updates. Change `max_step` or the damping slider and run again. Each run starts from the same selected start pose; playback does not change it.
2. Load **Near wrist singularity**. Compare damping 0.003 and 0.10, then the pseudoinverse. Look at the residual, trajectory and amplification gains together. These are iteration paths, not time-parameterized robot trajectories.
3. Load **Wrist singularity**. At wrist 2 = 0°, full-pose rank drops to 5. Switch to **Position only**: this preset still has position rank 3. Orientation residual remains visible but is not controlled in this task.
4. Load **Straight elbow**. Inspect the full-pose rank, then bend the elbow a little with its slider. A rank-deficient local Jacobian does not imply that all nearby targets are impossible. The provided reachable targets can be solved by leaving the singular configuration.
5. Select **Out-of-reach target**. A returned joint vector is independently measured and reported as “Residual remains” if it misses tolerance. Stop can terminate infinite loops; a 10-second limit also protects the interface.

Moving a joint makes the displayed pose the new start and keeps the target fixed. **Use this pose as target** captures the displayed TCP; move the joints afterward to create a new start. The example loader asks before replacing edited code.

## What runs where

- The page is plain static HTML, CSS and JavaScript, hosted by GitHub Pages.
- A Web Worker runs **Pyodide 0.29.3 + NumPy**, including the student's actual Python update rule. Native Pinocchio is not compiled into the browser.
- **Three.js 0.180.0** renders the mesh robot, target axes, TCP and recorded path. The MuJoCo-style floor and lighting are a visualization; no MuJoCo physics engine is running here.
- Native **Pinocchio 3.9.0** independently generated the checked-in reference data from the numerical URDF. Both the Python and JavaScript implementations are checked against these values.

The classroom runtime is intended for code the student chooses to run. It is not a security sandbox for untrusted third-party programs. Drafts are local to this site and browser; there is no shared server execution or student account system.

## Model and conventions

The arm is the **UR3 CB3**, with nominal uncalibrated geometry from official `ur_description` 2.9.0. It is not UR3e. The gripper is the **Robotiq 2F-85**, mounted with its official 11 mm adapter and fixed in its fully open posture. The adapter, fingers and gripper-body meshes all come from the official `2f_85` directory.

`classroom_tcp` is the midpoint of the area-weighted centres of the two inward fingertip contact faces in the fully open posture, oriented with the gripper base. The official 2F-85 model has no separate pad-contact or TCP frames, so the centres are derived from its actual CAD faces. In gripper-base coordinates the TCP is approximately `[0, -0.0003750023, 0.1302245943]` m; including the 11 mm adapter gives `[0, -0.0003750023, 0.1412245943]` m from `wrist_3_link`, with the same orientation. The small Y offset comes from the source CAD. The open pad gap is approximately 85 mm along the gripper-base X axis. This is a classroom TCP convention, not a calibrated tool definition. All targets and geometric Jacobians are in the model's world/base frame. Orientation inputs use `Rz(yaw) Ry(pitch) Rx(roll)`.

Joint order is shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3. Angles inside the solver are radians. The model's joint intervals are retained except for physically continuous wrist 3, where ±2π is an explicit numerical search interval, not a hardware limit. There is no collision detection, gripper closure, contact simulation, dynamics or physical robot connection.

`kinematics(q)` returns position, rotation and the world-aligned geometric Jacobian with linear rows first. The helper `task(q,target)` returns the selected residual and its effective correction Jacobian:

```python
# Position residual and geometric diagnostic Jacobian
e_position = (p_target - p) / 0.4
J_position = J_linear / 0.4

# Full pose adds a rotation-vector residual
e_rotation = log_SO3(R_target @ R.T)
J_rotation = right_log_jacobian_inverse(e_rotation) @ J_angular
```

The solver's orientation Jacobian includes the derivative of the log residual. The singularity lens instead measures the **geometric Jacobian**, independent of target. Its full-pose scaling is `[J_linear / L; J_angular]`, with characteristic length `L = 0.4 m`. Scaling changes condition estimates but not exact rank. The display treats singular values below `10^-7 σmax` as numerically zero; “near” means `σmin / σmax < 0.01`. The pseudoinverse template uses its own `rcond=10^-6`. DLS weights a singular direction by `σ/(σ²+λ²)`. These displayed gains are before step limiting and backtracking; damping cannot restore a lost direction.

The 3D ellipsoid uses position singular vectors and normalized singular-value ratios only. It cannot visualize six-dimensional pose manipulability. The wrist sweep changes only wrist 2, holds the other joints at the displayed pose, and floors logarithmic plot values at `10^-12` for plotting. Solver tolerances are 0.1 mm position and, in full-pose mode, 0.1° orientation.

## Development and validation

No build or package installation is required to serve the website:

```bash
python3 -m http.server 8771 --directory dist
# Open http://localhost:8771
node qa/check.mjs
python3 -m pip install numpy==2.2.6
python3 qa/check_python.py
```

The workflow validates JavaScript and Python before publishing `dist/`. Checks cover 20 independent Pinocchio FK/Jacobian references, NumPy SVD values, renderer/tool alignment, finite-difference derivatives, singularity ranks, solver convergence, rejected invalid outputs and an unreachable target. `qa/browser.py` additionally exercises actual Pyodide, editor errors, Stop/recovery, playback, exports and responsive layouts through a separately started Chrome DevTools session on port 9340. Its screenshots are local QA artifacts, not deployment assets.

To regenerate the independent reference fixture, use an environment with native Pinocchio and NumPy and run `python qa/generate_reference.py`. This is a maintainer step; students do not need Pinocchio installed.

All runtime mesh and model assets are already checked in. The optional `scripts/build_robot_assets.py --ur-source PATH` is a maintainer conversion tool using NumPy, SciPy and xacro. It expects an existing UR3 conversion directory containing `source.urdf`, `source_wrapper.xacro`, `robot.xml`, the referenced material-split OBJ meshes, `provenance.json` and `LICENSE`. That intermediate UR conversion directory is not included; hashes and upstream version are recorded in the published provenance. The Robotiq sources used for assembly are included under `models/robotiq_source/`. Rebuilding assets is not needed for testing, editing solvers or deployment.

## Sources and licenses

- [Universal Robots ROS 2 Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description), version 2.9.0. The full [UR license and graphical-documentation terms](dist/assets/robot/UR_LICENSE.txt) are retained. **The graphical assets are not covered by a blanket BSD license.**
- [Official Robotiq sources](https://github.com/robotiq/ros/tree/7d0bc4aeaf16c8934520667bac4617e192a866ce/grippers/robotiq_description), pinned commit `7d0bc4aeaf16c8934520667bac4617e192a866ce`; [BSD-3-Clause license](dist/assets/robot/ROBOTIQ_LICENSE.txt).
- [Three.js](https://threejs.org/), with its MIT license in `dist/vendor/`.
- [Pyodide](https://pyodide.org/en/stable/), loaded from the versioned CDN by `dist/python-worker.js`.
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio), used for native reference validation.

Detailed source hashes, mounting information and the mesh checksum are in [provenance.json](dist/assets/robot/provenance.json). The inherited UR conversion's collision/inertia metadata describes its source artifact; this web exercise uses visual geometry and kinematics only.

This is an independent classroom lab. The existing [G1 lecture](https://miguelaltamirano.github.io/g1-pinocchio-lab/) and [G1 Python playground](https://miguelaltamirano.github.io/g1-ik-playground/) remain separate.

The earlier 2F-140 version is retained in Git tag `v1-2f140`. Existing browser code drafts are migrated when opening the updated lab.
