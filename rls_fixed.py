r"""
RLS FORCE baseline + offline ridge regression upper bound.

Two baselines:
  1. Offline Ridge Regression - theoretical upper bound for linear readout
  2. Online RLS (FORCE) - with heavy numerical stabilization

Run with: D:\EGO\run_rls_fix.bat
"""
import numpy as np
import time
import sys
import traceback

# NOTE: only truncate the results file when run as a script.
# Importing this module must NOT clobber rls_fix_results.txt.
if __name__ == '__main__':
    _logfile = open('rls_fix_results.txt', 'w', encoding='utf-8')
else:
    import io
    _logfile = io.StringIO()

def out(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass
    _logfile.write(msg + '\n')
    _logfile.flush()

out("RLS + Ridge Baseline Experiment")
out("=" * 60)

try:
    from scipy import stats as scipy_stats
    out("scipy: available")
except ImportError:
    scipy_stats = None
    out("scipy: not available")


# ============================================================
#  Components
# ============================================================

class SpikingReservoir:
    def __init__(self, n=1000, n_input=200, seed=42,
                 tau_m=10.0, v_th=1.0, v_reset=0.0, t_ref=2.0,
                 g=5.0, sparseness=0.1, dt=1.0,
                 bias=1.2, noise_std=0.3, w_in_scale=0.5):
        self.n = n
        self.n_input = n_input
        self.dt = dt
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_reset = v_reset
        self.t_ref = t_ref
        self.rng = np.random.RandomState(seed)
        self.v = self.rng.uniform(0, v_th, n).astype(np.float32)
        self.refractory = np.zeros(n, dtype=np.float32)
        self.fired = np.zeros(n, dtype=np.float32)
        p = sparseness
        scale = g / np.sqrt(p * n)
        mask = self.rng.random((n, n)) < p
        self.w_rec = np.zeros((n, n), dtype=np.float32)
        self.w_rec[mask] = self.rng.randn(mask.sum()).astype(np.float32) * scale
        if n_input > 0:
            self.w_in = self.rng.randn(n, n_input).astype(np.float32) * w_in_scale
        else:
            self.w_in = None
        self.bias = bias
        self.noise_std = noise_std

    def step(self, input_spikes=None):
        self.refractory = np.maximum(0.0, self.refractory - self.dt)
        I_rec = self.w_rec @ self.fired + self.bias
        if self.noise_std > 0:
            I_rec += self.rng.randn(self.n).astype(np.float32) * self.noise_std
        if input_spikes is not None and self.w_in is not None:
            I_rec += self.w_in @ input_spikes
        active = self.refractory <= 0.0
        dv = (-self.v + I_rec) / self.tau_m * self.dt
        self.v[active] += dv[active]
        self.v[self.v < self.v_reset] = self.v_reset
        self.fired = (self.v >= self.v_th).astype(np.float32)
        self.v[self.fired > 0.5] = self.v_reset
        self.refractory[self.fired > 0.5] = self.t_ref
        return self.fired.copy()


class VisualEncoder:
    def __init__(self, grid_h=20, grid_w=10, seed=42):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_vis = grid_h * grid_w
        self.rng = np.random.RandomState(seed)
        ii, jj = np.mgrid[0:grid_h, 0:grid_w]
        self._patterns = []
        for shape_id in range(3):
            grid = np.zeros((grid_h, grid_w), dtype=np.float32)
            if shape_id == 0:
                grid[2:14, 1:5] = 1.0
            elif shape_id == 1:
                cx, cy = 6, 7.5
                mask = ((ii - cx)**2 / 25 + (jj - cy)**2 / 6.25) <= 1
                grid[mask] = 1.0
            elif shape_id == 2:
                base_row = 19; tip_row = 7; base_half_w = 3.0
                center_col = 5.0; height = base_row - tip_row
                row_dist = base_row - ii
                width_at_row = (row_dist / height) * base_half_w
                dist_from_center = np.abs(jj - center_col)
                mask = (ii >= tip_row) & (ii <= base_row) & (dist_from_center <= width_at_row)
                grid[mask] = 1.0
            self._patterns.append(grid.flatten())

    def encode(self, shape_id, rate=50.0):
        rates = self._patterns[shape_id] * rate
        rates += self.rng.uniform(0, 2, self.n_vis).astype(np.float32)
        return (self.rng.random(self.n_vis) < rates / 1000.0).astype(np.float32)


class VocalTract:
    def __init__(self, n_params=8, seed=42):
        self.n = n_params
        self.rng = np.random.RandomState(seed)
        self.mass = np.ones(n_params, dtype=np.float32) * 1.0
        self.spring_k = np.ones(n_params, dtype=np.float32) * 2.0
        self.damping = np.ones(n_params, dtype=np.float32) * 2.5
        self.rest_pos = np.ones(n_params, dtype=np.float32) * 0.5
        self.position = self.rest_pos.copy()
        self.velocity = np.zeros(n_params, dtype=np.float32)

    def step(self, force, dt=1.0):
        spring_force = -self.spring_k * (self.position - self.rest_pos)
        damping_force = -self.damping * self.velocity
        total_force = force + spring_force + damping_force
        acceleration = total_force / self.mass
        self.velocity += acceleration * dt * 0.1
        np.clip(self.velocity, -10.0, 10.0, out=self.velocity)
        self.position += self.velocity * dt * 0.1
        self.position = np.clip(self.position, 0.0, 1.0)
        hit_wall = (self.position <= 0.01) | (self.position >= 0.99)
        self.velocity[hit_wall] *= -0.3
        return self.position.copy()

    def reset(self):
        self.position = self.rest_pos.copy()
        self.velocity = np.zeros(self.n, dtype=np.float32)


def generate_target_trajectory(shape_id, duration_steps=1000):
    t = np.linspace(0, 1, duration_steps, dtype=np.float32)
    params = np.zeros((duration_steps, 8), dtype=np.float32)
    if shape_id == 0:
        params[:, 0] = 0.5 + 0.15 * np.sin(2*np.pi*3*t)
        params[:, 1] = 0.5 + 0.10 * np.sin(2*np.pi*2*t)
        params[:, 2] = 0.6; params[:, 3] = 0.5; params[:, 4] = 0.1
        params[:, 5] = 0.5 + 0.20 * np.sin(2*np.pi*5*t)
        params[:, 6] = 0.05; params[:, 7] = 0.5 + 0.10*np.sin(2*np.pi*1*t)
    elif shape_id == 1:
        params[:, 0] = 0.5 + 0.20*(1-np.cos(np.pi*t))
        params[:, 1] = 0.5 + 0.15*t; params[:, 2] = 0.5 + 0.10*t
        params[:, 3] = 0.5; params[:, 4] = 0.05
        params[:, 5] = 0.5 + 0.30*np.sin(np.pi*t)
        params[:, 6] = 0.05; params[:, 7] = 0.5
    elif shape_id == 2:
        params[:, 0] = 0.5 + 0.20*np.sin(2*np.pi*8*t)
        params[:, 1] = 0.5 + 0.10*np.sin(2*np.pi*4*t)
        params[:, 2] = 0.5 + 0.15*np.sin(2*np.pi*8*t)
        params[:, 3] = 0.5; params[:, 4] = 0.15
        params[:, 5] = 0.5 + 0.15*np.sin(2*np.pi*2*t)
        params[:, 6] = 0.08; params[:, 7] = 0.5 + 0.20*np.sin(2*np.pi*4*t)
    params = np.clip(params, 0, 1)
    return params


def compute_target_force(position, velocity, target_pos, kp=5.0, kd=2.0):
    error = target_pos - position
    force = error * kp - velocity * kd
    return force.astype(np.float32)


def pearson_corr(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    if np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def precompute_target_forces(target_pos_traj, kp=5.0, kd=2.0):
    """Simulate perfect tracking to get target force trajectory."""
    n_steps = len(target_pos_traj)
    n_params = target_pos_traj.shape[1]
    vt = VocalTract(n_params=n_params, seed=0)
    forces = np.zeros((n_steps, n_params), dtype=np.float32)
    for i in range(n_steps):
        target_pos = target_pos_traj[i]
        force = compute_target_force(vt.position, vt.velocity, target_pos, kp, kd)
        forces[i] = force
        vt.step(force)
    return forces


# ============================================================
#  PSP filter helper
# ============================================================

def psp_filter_spikes(spike_train, tau_psp, dt):
    """
    Apply exponential PSP filter to a spike train matrix.
    spike_train: (T, N) binary matrix
    Returns: (T, N) filtered PSP values, normalized by tau/dt
    """
    T, N = spike_train.shape
    decay = np.exp(-dt / tau_psp)
    psp = np.zeros((T, N), dtype=np.float64)
    state = np.zeros(N, dtype=np.float64)
    norm = tau_psp / dt  # normalize to firing-rate range
    for t in range(T):
        state = state * decay + spike_train[t].astype(np.float64)
        psp[t] = state / norm
    return psp


# ============================================================
#  Offline Ridge Regression (theoretical upper bound)
# ============================================================

def ridge_regression(X, Y, alpha=1.0):
    """
    Solve Y = W @ X.T  =>  W = Y @ X.T @ (X @ X.T + alpha*I)^(-1)
    X: (T, N) feature matrix (PSP-filtered reservoir states)
    Y: (T, M) target matrix
    alpha: regularization parameter
    Returns: W (M, N)
    """
    T, N = X.shape
    # Use the trick: if T < N, solve in T-dimensional space
    # X @ X.T is (T, T) instead of (N, N)
    # W = Y.T @ X @ (X.T @ X + alpha*I)^(-1)  ... no, let me think
    # We want: W (M x N) such that Y ≈ W @ X.T  (M x T)
    # Solution: W = Y @ X @ (X.T @ X + alpha*I)^(-1)
    # For large N but maybe not too large T, use dual form:
    # W = Y @ (X @ X.T + alpha*I_T)^(-1) @ X   No...
    
    # Standard form: W = Y.T @ X @ inv(X.T @ X + alpha*I)
    # Let's just use np.linalg.lstsq with augmented system
    X_aug = np.vstack([X, np.sqrt(alpha) * np.eye(N)])
    Y_aug = np.vstack([Y, np.zeros((N, Y.shape[1]))])
    W, residuals, rank, sv = np.linalg.lstsq(X_aug, Y_aug, rcond=None)
    return W.T  # (M, N)


# ============================================================
#  Collect reservoir states (for offline ridge)
# ============================================================

def collect_reservoir_states(reservoir, vis_enc, shape_id, n_steps, tau_psp, dt):
    """Run reservoir and collect PSP-filtered states + target forces."""
    # First generate spike train
    spikes = np.zeros((n_steps, reservoir.n), dtype=np.float32)
    for t in range(n_steps):
        vis_spikes = vis_enc.encode(shape_id, rate=50.0)
        fired = reservoir.step(vis_spikes)
        spikes[t] = fired
    
    # PSP filter
    psp = psp_filter_spikes(spikes, tau_psp, dt)
    return psp.astype(np.float64)


# ============================================================
#  Test readout with vocal tract (closed loop)
# ============================================================

def test_readout_closedloop(reservoir, w_out, vis_enc, vt, targets_pos, shapes,
                            test_steps=500, tau_psp=100.0, dt=1.0, force_gain=3.0):
    """Test a linear readout (W_out) in closed loop with vocal tract."""
    shape_names = ['square', 'circle', 'triangle']
    decay = np.exp(-dt / tau_psp)
    psp_norm = tau_psp / dt
    results = {}
    
    for shape_id in shapes:
        vt.reset()
        psp_state = np.zeros(reservoir.n, dtype=np.float64)
        outputs = []
        targets_arr = []
        for step in range(test_steps):
            vis_spikes = vis_enc.encode(shape_id, rate=50.0)
            fired = reservoir.step(vis_spikes)
            psp_state = psp_state * decay + fired.astype(np.float64)
            x = psp_state / psp_norm
            output = w_out @ x
            force = output * force_gain
            np.clip(force, -20.0, 20.0, out=force)
            vt.step(force.astype(np.float32))
            outputs.append(vt.position.copy())
            targets_arr.append(targets_pos[shape_id][step % len(targets_pos[shape_id])].copy())
        outputs = np.array(outputs)
        targets_arr = np.array(targets_arr)
        corr = pearson_corr(outputs, targets_arr)
        results[shape_names[shape_id]] = corr
    return results


# ============================================================
#  Online RLS (FORCE) - heavily stabilized
# ============================================================

class RLSOnline:
    """
    Online RLS with heavy stabilization for spiking networks.
    Uses float64 internally, P matrix clipping, strong regularization.
    """
    def __init__(self, n_reservoir, n_output=8, seed=42,
                 alpha=100.0, lam=1.0, tau_psp=20.0, dt=1.0,
                 p_max=1e6, w_max=10.0):
        self.n_in = n_reservoir
        self.n_out = n_output
        self.lam = lam
        self.dt = dt
        self.tau_psp = tau_psp
        self.p_max = p_max
        self.w_max = w_max
        self.rng = np.random.RandomState(seed)
        self.w_out = (self.rng.randn(n_output, n_reservoir) * 0.01).astype(np.float64)
        self.P = np.eye(n_reservoir, dtype=np.float64) * alpha
        self.psp = np.zeros(n_reservoir, dtype=np.float64)
        self.decay = np.exp(-dt / tau_psp)
        self.psp_norm = tau_psp / dt

    def train_step(self, reservoir_fired, target):
        self.psp = self.psp * self.decay + reservoir_fired.astype(np.float64)
        x = self.psp / self.psp_norm
        z = self.w_out @ x
        e = target.astype(np.float64) - z

        Px = self.P @ x
        denom = self.lam + x @ Px
        if denom < 1e-8:
            denom = 1e-8
        k = Px / denom
        self.P = (self.P - np.outer(k, x @ self.P)) / self.lam
        # Symmetrize
        self.P = (self.P + self.P.T) * 0.5
        # Clip P to prevent blowup
        np.clip(self.P, -self.p_max, self.p_max, out=self.P)
        # Weight update
        self.w_out += np.outer(e, k)
        np.clip(self.w_out, -self.w_max, self.w_max, out=self.w_out)

        return z, e

    def get_output(self, reservoir_fired):
        self.psp = self.psp * self.decay + reservoir_fired.astype(np.float64)
        x = self.psp / self.psp_norm
        return self.w_out @ x


def train_rls_online(reservoir, rls, vis_enc, targets_force, shapes, train_steps, n_rounds=3):
    steps_per_block = train_steps // (n_rounds * len(shapes))
    for rnd in range(n_rounds):
        for shape_id in shapes:
            force_traj = targets_force[shape_id]
            for step in range(steps_per_block):
                target_idx = step % len(force_traj)
                vis_spikes = vis_enc.encode(shape_id, rate=50.0)
                fired = reservoir.step(vis_spikes)
                target_force = force_traj[target_idx]
                rls.train_step(fired, target_force)


# ============================================================
#  Main
# ============================================================

def main():
    out("")
    out("=" * 60)
    out("  BASELINE COMPARISON: Ridge (offline) + RLS (online)")
    out("=" * 60)
    out("")
    out("Part A: Offline Ridge Regression (theoretical upper bound)")
    out("  - Answers: can a linear readout solve this task at all?")
    out("  - Uses full training data, no online constraint")
    out("")
    out("Part B: Online RLS / FORCE (stabilized)")
    out("  - Answers: can online RLS converge on spiking reservoir?")
    out("")

    t0 = time.time()
    n_seeds = 10
    seeds = list(range(n_seeds))
    targets_pos = [generate_target_trajectory(s, 1000) for s in range(3)]

    # === Part A: Offline Ridge ===
    out("--- Part A: Offline Ridge Regression ---")
    tau_values = [20, 50, 100, 200]
    ridge_results = {}
    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    for tau in tau_values:
        best_corrs = []  # best alpha per seed
        best_alpha_per_seed = []
        t_tau = time.time()
        out(f"  tau_psp = {tau} ms:")

        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed + 200)
            
            # Precompute target forces
            targets_force = [precompute_target_forces(targets_pos[s]) for s in range(3)]
            
            # Collect training data (same amount as online: 9000 steps total = 3000 per shape)
            train_steps_per_shape = 3000
            all_psp = []
            all_targets = []
            
            for shape_id in [0, 1, 2]:
                # Reset reservoir state for clean collection
                # (we just run and collect; reservoir dynamics evolve naturally)
                psp = collect_reservoir_states(
                    reservoir, vis_enc, shape_id, train_steps_per_shape, tau, 1.0)
                force_traj = targets_force[shape_id]
                # Repeat target trajectory as needed
                n_repeats = train_steps_per_shape // len(force_traj) + 1
                target_repeated = np.tile(force_traj, (n_repeats, 1))[:train_steps_per_shape]
                all_psp.append(psp)
                all_targets.append(target_repeated.astype(np.float64))
            
            X_train = np.vstack(all_psp)  # (9000, 1000)
            Y_train = np.vstack(all_targets)  # (9000, 8)
            
            # Try different regularization strengths, pick best on training fit
            best_corr = -1
            best_w = None
            best_alpha = None
            
            for alpha in ridge_alphas:
                try:
                    w_ridge = ridge_regression(X_train, Y_train, alpha=alpha)
                    # Quick check: training correlation
                    Y_pred = X_train @ w_ridge.T
                    train_corr = pearson_corr(Y_pred, Y_train)
                    if train_corr > best_corr:
                        best_corr = train_corr
                        best_w = w_ridge
                        best_alpha = alpha
                except Exception:
                    continue
            
            # Test closed-loop with best weights
            # Need a fresh reservoir (same seed) for fair test
            reservoir_test = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            results = test_readout_closedloop(
                reservoir_test, best_w, vis_enc, vt, targets_pos, [0, 1, 2],
                test_steps=500, tau_psp=tau, dt=1.0, force_gain=3.0)
            avg = np.mean([results[s] for s in ['square', 'circle', 'triangle']])
            best_corrs.append(avg)
            best_alpha_per_seed.append(best_alpha)
            out(f"    seed {seed:2d}: {avg:7.3f}  (best alpha={best_alpha})")

        mean_v = np.mean(best_corrs)
        std_v = np.std(best_corrs)
        ridge_results[tau] = best_corrs
        elapsed = time.time() - t_tau
        out(f"    >>> tau={tau:3d}ms: {mean_v:.3f} +/- {std_v:.3f}  ({elapsed:.0f}s)")
        out("")

    # === Part B: Online RLS ===
    out("--- Part B: Online RLS (stabilized) ---")
    rls_results = {}
    
    for tau in tau_values:
        corrs = []
        t_tau = time.time()
        out(f"  tau_psp = {tau} ms:")
        
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed + 200)
            targets_force = [precompute_target_forces(targets_pos[s]) for s in range(3)]
            
            rls = RLSOnline(1000, 8, seed=seed + 100,
                            alpha=100.0, lam=1.0, tau_psp=float(tau),
                            p_max=1e6, w_max=10.0)
            
            try:
                train_rls_online(reservoir, rls, vis_enc, targets_force,
                                 [0, 1, 2], train_steps=9000)
                
                # Test (fresh reservoir)
                reservoir_test = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
                results = test_readout_closedloop(
                    reservoir_test, rls.w_out.astype(np.float64), vis_enc, vt,
                    targets_pos, [0, 1, 2],
                    test_steps=500, tau_psp=tau, dt=1.0, force_gain=3.0)
                avg = np.mean([results[s] for s in ['square', 'circle', 'triangle']])
                
                if np.isnan(avg) or np.isinf(avg):
                    avg = 0.0
                    out(f"    seed {seed:2d}: FAILED (diverged)")
                else:
                    out(f"    seed {seed:2d}: {avg:7.3f}")
                corrs.append(avg)
            except Exception as e:
                out(f"    seed {seed:2d}: ERROR ({str(e)[:50]})")
                corrs.append(0.0)
        
        mean_v = np.mean(corrs)
        std_v = np.std(corrs)
        rls_results[tau] = corrs
        elapsed = time.time() - t_tau
        out(f"    >>> tau={tau:3d}ms: {mean_v:.3f} +/- {std_v:.3f}  ({elapsed:.0f}s)")
        out("")

    # === Final Summary ===
    total_time = time.time() - t0
    out("=" * 60)
    out("  FINAL SUMMARY")
    out(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    out("=" * 60)
    out("")
    out("  [A] Offline Ridge Regression (theoretical upper bound):")
    for tau, corrs in ridge_results.items():
        out(f"    tau={tau:3d}ms: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")
    out("")
    out("  [B] Online RLS / FORCE (stabilized):")
    for tau, corrs in rls_results.items():
        out(f"    tau={tau:3d}ms: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")
    out("")
    out("  Reference: STDP + physical guidance (membrane) = 0.865 +/- 0.022")
    out("")

    # Statistical comparison
    stdp_results = np.array([0.880, 0.863, 0.885, 0.847, 0.827, 0.849, 0.901, 0.843, 0.876, 0.877])
    out("  Comparison with STDP (best tau per method):")
    best_ridge_tau = max(ridge_results.keys(), key=lambda t: np.mean(ridge_results[t]))
    best_rls_tau = max(rls_results.keys(), key=lambda t: np.mean(rls_results[t]))
    
    for name, tau, results_dict in [("Ridge (offline)", best_ridge_tau, ridge_results),
                                     ("RLS (online)", best_rls_tau, rls_results)]:
        arr = np.array(results_dict[tau])
        if scipy_stats is not None:
            t_stat, p_val = scipy_stats.ttest_ind(stdp_results, arr)
        else:
            t_stat, p_val = 0.0, 1.0
        pooled = np.sqrt((np.std(stdp_results)**2 + np.std(arr)**2) / 2)
        d = (np.mean(stdp_results) - np.mean(arr)) / pooled if pooled > 0 else 0
        out(f"    {name} (tau={tau}ms): {np.mean(arr):.3f}, t={t_stat:.2f}, p={p_val:.2e}, d={d:.2f}")
    
    out("")
    out("  === EXPERIMENT COMPLETE ===")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        _logfile.write("\nERROR:\n" + err + "\n")
        _logfile.flush()
        try:
            print("\nERROR:", flush=True)
            print(err, flush=True)
        except Exception:
            pass
    finally:
        _logfile.write("\n=== SCRIPT ENDED ===\n")
        _logfile.close()
