# Provided: np, task, converged, record, lower, upper, DAMPING.
# task(q, target) returns a scaled error and its task Jacobian.
def solve_ik(q, target):
    max_step = 0.12

    for iteration in range(400):
        error, J = task(q, target)
        if converged(q, target):
            break

        # Try changing this update rule.
        dq = np.linalg.pinv(J, rcond=1e-6) @ error
        dq *= min(1.0, max_step / max(np.linalg.norm(dq), 1e-15))

        for trial in range(15):
            candidate = np.clip(q + 0.5**trial * dq, lower, upper)
            new_error, _ = task(candidate, target)
            if np.linalg.norm(new_error) < np.linalg.norm(error) - 1e-13:
                q = candidate
                record(q)
                break
        else:
            print('Stopped: no improving step.')
            break

    print('Final joints (degrees):', np.round(np.rad2deg(q), 2))
    return q
