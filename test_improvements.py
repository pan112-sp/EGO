r"""
Paper revision tests - standalone version.
No external imports, no dependencies beyond numpy/scipy.

How to run:
    cd D:\EGO
    C:\Users\Pan haimeng\AppData\Local\Programs\Python\Python312\python.exe test_improvements.py
"""
import numpy as np
import time
import sys
import traceback

_f = open('improvements_results.txt', 'w', encoding='utf-8')

def _out(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass
    _f.write(msg + '\n')
    _f.flush()

_f.write("Script started.\n")
_f.flush()

try:
    from scipy import stats as scipy_stats
    _f.write("scipy imported OK\n")
    _f.flush()
except ImportError:
    _f.write("scipy not available, using manual stats\n")
    _f.flush()
    scipy_stats = None


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
        self.rate_ema = 0.0
        self.rate_alpha = 0.999
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
        n_spikes = self.fired.sum()
        self.rate_ema = self.rate_alpha * self.rate_ema + (1 - self.rate_alpha) * n_spikes
        return self.fired.copy()


class STDPReadout:
    def __init__(self, n_reservoir, n_output=8, seed=42,
                 w_max=5.0, a_plus=0.01, a_minus=0.01,
                 tau_stdp=20.0, tau_elig=100.0,
                 homeo_target=10.0, homeo_gain=0.0001,
                 dt=1.0,
                 use_stdp=True, use_eligibility=True, use_homeostasis=True):
        self.n_in = n_reservoir
        self.n_out = n_output
        self.dt = dt
        self.w_max = w_max
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_stdp = tau_stdp
        self.tau_elig = tau_elig
        self.homeo_target = homeo_target
        self.homeo_gain = homeo_gain
        self.use_stdp = use_stdp
        self.use_eligibility = use_eligibility
        self.use_homeostasis = use_homeostasis
        self.rng = np.random.RandomState(seed)
        self.w_out = self.rng.rand(n_output, n_reservoir).astype(np.float32) * 0.1
        self.x_pre = np.zeros(n_reservoir, dtype=np.float32)
        self.y_post = np.zeros(n_output, dtype=np.float32)
        self.elig = np.zeros((n_output, n_reservoir), dtype=np.float32)
        self.v_out = np.zeros(n_output, dtype=np.float32)
        self.fired_out = np.zeros(n_output, dtype=np.float32)
        self.refractory_out = np.zeros(n_output, dtype=np.float32)
        self.rate_out = np.zeros(n_output, dtype=np.float32)

    def step(self, reservoir_fired, target_current=None):
        decay_pre = np.exp(-self.dt / self.tau_stdp).astype(np.float32)
        self.x_pre = self.x_pre * decay_pre + reservoir_fired.astype(np.float32)
        self.y_post = self.y_post * decay_pre + self.fired_out.astype(np.float32)
        if self.use_eligibility:
            decay_elig = np.exp(-self.dt / self.tau_elig).astype(np.float32)
            self.elig *= decay_elig
            co_activity = np.outer(self.fired_out.astype(np.float32),
                                   reservoir_fired.astype(np.float32))
            self.elig += co_activity
        self.refractory_out = np.maximum(0.0, self.refractory_out - self.dt)
        I_syn = self.w_out @ reservoir_fired.astype(np.float32)
        if target_current is not None:
            I_syn = I_syn + target_current
        active = self.refractory_out <= 0.0
        tau_out = 10.0
        dv = (-self.v_out + I_syn) / tau_out * self.dt
        self.v_out[active] += dv[active]
        self.v_out[self.v_out < 0] = 0
        self.fired_out = (self.v_out >= 1.0).astype(np.float32)
        self.v_out[self.fired_out > 0.5] = 0.0
        self.refractory_out[self.fired_out > 0.5] = 2.0
        if self.use_homeostasis:
            self.rate_out = 0.999 * self.rate_out + 0.001 * self.fired_out
            rate_error = self.rate_out - self.homeo_target / 1000.0
            homeo_adj = -self.homeo_gain * rate_error[:, np.newaxis]
            self.w_out += homeo_adj.astype(np.float32)
        if self.use_stdp and target_current is not None:
            ltp = np.outer(self.fired_out.astype(np.float32), self.x_pre)
            ltd = np.outer(self.y_post, reservoir_fired.astype(np.float32))
            delta_w = (self.a_plus * ltp - self.a_minus * ltd).astype(np.float32)
            self.w_out += delta_w
        np.clip(self.w_out, 0.0, self.w_max, out=self.w_out)
        return self.fired_out.copy()

    def get_output(self, reservoir_fired):
        raw = self.w_out @ reservoir_fired.astype(np.float32)
        centered = raw - raw.mean()
        return centered


class STDPReadoutRate(STDPReadout):
    def __init__(self, n_reservoir, n_output=8, seed=42,
                 rate_window=100, rate_gain=20.0, **kwargs):
        super().__init__(n_reservoir, n_output, seed, **kwargs)
        self.rate_window = rate_window
        self.rate_gain = rate_gain
        self.spike_history = np.zeros((rate_window, n_output), dtype=np.float32)
        self.hist_idx = 0

    def step(self, reservoir_fired, target_current=None):
        result = super().step(reservoir_fired, target_current)
        self.spike_history[self.hist_idx] = self.fired_out
        self.hist_idx = (self.hist_idx + 1) % self.rate_window
        return result

    def get_output(self, reservoir_fired):
        spike_count = self.spike_history.sum(axis=0).astype(np.float32)
        centered = spike_count - spike_count.mean()
        return centered * self.rate_gain


class RLSReadout:
    def __init__(self, n_reservoir, n_output=8, seed=42,
                 alpha=1.0, lam=0.99, tau_psp=20.0, dt=1.0):
        self.n_in = n_reservoir
        self.n_out = n_output
        self.lam = lam
        self.dt = dt
        self.tau_psp = tau_psp
        self.rng = np.random.RandomState(seed)
        self.w_out = self.rng.randn(n_output, n_reservoir).astype(np.float32) * 0.01
        self.P = np.eye(n_reservoir, dtype=np.float32) * alpha
        self.P_reg = 1e-6
        self.psp_filter = np.zeros(n_reservoir, dtype=np.float32)
        self.decay_psp = np.exp(-dt / tau_psp).astype(np.float32)

    def step(self, reservoir_fired, target_current=None):
        self.psp_filter = self.psp_filter * self.decay_psp + reservoir_fired.astype(np.float32)

    def train_rls(self, reservoir_fired, target):
        self.psp_filter = self.psp_filter * self.decay_psp + reservoir_fired.astype(np.float32)
        x = self.psp_filter.copy()
        z = self.w_out @ x
        e = target - z
        Px = self.P @ x
        denom = self.lam + x @ Px
        if denom < 1e-10:
            denom = 1e-10
        k = Px / denom
        self.P = (self.P - np.outer(k, x @ self.P)) / self.lam
        self.P += np.eye(self.n_in, dtype=np.float32) * self.P_reg
        np.clip(self.P, -1e6, 1e6, out=self.P)
        delta_w = np.outer(e, k)
        np.clip(delta_w, -1.0, 1.0, out=delta_w)
        self.w_out += delta_w.astype(np.float32)
        return z, e

    def get_output(self, reservoir_fired):
        return self.w_out @ self.psp_filter


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
    def __init__(self, n_params=8, resistance=0.5, seed=42):
        self.n = n_params
        self.rng = np.random.RandomState(seed)
        self.mass = np.ones(n_params, dtype=np.float32) * 1.0
        self.spring_k = np.ones(n_params, dtype=np.float32) * 2.0
        self.damping = np.ones(n_params, dtype=np.float32) * (1.0 + resistance * 3.0)
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


def train_stdp(reservoir, readout, vis_enc, vt, targets, shapes,
               train_steps, teach_current, n_rounds=3):
    steps_per_block = train_steps // (n_rounds * len(shapes))
    for rnd in range(n_rounds):
        for shape_id in shapes:
            target = targets[shape_id]
            vt.reset()
            for step in range(steps_per_block):
                target_idx = step % len(target)
                vis_spikes = vis_enc.encode(shape_id, rate=50.0)
                fired = reservoir.step(vis_spikes)
                target_force = compute_target_force(
                    vt.position, vt.velocity, target[target_idx])
                teach_curr = target_force * teach_current
                readout.step(fired, target_current=teach_curr)
                output = readout.get_output(fired)
                total_progress = (rnd * len(shapes) + shape_id) * steps_per_block + step
                mix = min(1.0, total_progress / (train_steps * 0.5))
                mixed_force = (1 - mix) * target_force + mix * output / 3.0
                vt.step(mixed_force)


def train_rls(reservoir, readout, vis_enc, vt, targets, shapes,
              train_steps, n_rounds=3):
    steps_per_block = train_steps // (n_rounds * len(shapes))
    for rnd in range(n_rounds):
        for shape_id in shapes:
            target = targets[shape_id]
            vt.reset()
            for step in range(steps_per_block):
                target_idx = step % len(target)
                vis_spikes = vis_enc.encode(shape_id, rate=50.0)
                fired = reservoir.step(vis_spikes)
                target_force = compute_target_force(
                    vt.position, vt.velocity, target[target_idx])
                readout.train_rls(fired, target_force)
                output = readout.get_output(fired)
                vt.step(output * 3.0)


def test_free_run(reservoir, readout, vis_enc, vt, targets, shapes, test_steps=500):
    results = {}
    shape_names = ['square', 'circle', 'triangle']
    for shape_id in shapes:
        vt.reset()
        outputs = []
        targets_arr = []
        for step in range(test_steps):
            vis_spikes = vis_enc.encode(shape_id, rate=50.0)
            fired = reservoir.step(vis_spikes)
            readout.step(fired, target_current=None)
            output = readout.get_output(fired)
            force = output * 3.0 / 10.0
            np.clip(force, -20.0, 20.0, out=force)
            vt.step(force)
            outputs.append(vt.position.copy())
            targets_arr.append(targets[shape_id][step % len(targets[shape_id])].copy())
        outputs = np.array(outputs)
        targets_arr = np.array(targets_arr)
        corr = pearson_corr(outputs, targets_arr)
        results[shape_names[shape_id]] = corr
    return results


# ============================================================
#  Main
# ============================================================
def main():
    _out("")
    _out("=" * 60)
    _out("  EGO Paper Revision Tests")
    _out("  A: RLS PSP sweep (20/50/100ms, 10 seeds each)")
    _out("  B: Spike-rate vs membrane readout (10 seeds each)")
    _out("=" * 60)

    t0 = time.time()
    seeds = list(range(10))

    # === Experiment A: RLS PSP Sweep ===
    _out("\n--- Experiment A: RLS PSP Time Constant Sweep ---")
    tau_values = [20, 50, 100]
    rls_results = {}

    for tau in tau_values:
        corrs = []
        t_tau = time.time()
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed + 200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]
            readout = RLSReadout(1000, 8, seed=seed + 100, tau_psp=float(tau))
            train_rls(reservoir, readout, vis_enc, vt, targets, [0, 1, 2], 9000)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0, 1, 2])
            avg = np.mean([results[s] for s in ['square', 'circle', 'triangle']])
            corrs.append(avg)
            _out(f"  tau={tau:3d}ms seed {seed}: {avg:.3f}")
        mean_v = np.mean(corrs)
        std_v = np.std(corrs)
        rls_results[tau] = corrs
        elapsed = time.time() - t_tau
        _out(f"  >>> tau={tau:3d}ms: {mean_v:.3f} +/- {std_v:.3f} ({elapsed:.0f}s)")

    # === Experiment B: Readout Comparison ===
    _out("\n--- Experiment B: Spike-Rate vs Membrane Readout ---")
    readout_results = {}

    configs = [
        ('Membrane (original)', 'membrane'),
        ('Spike-rate w=100', 'rate100'),
        ('Spike-rate w=50', 'rate50'),
    ]

    for name, cfg in configs:
        corrs = []
        t_cfg = time.time()
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed + 200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]

            if cfg == 'membrane':
                readout = STDPReadout(1000, 8, seed=seed + 100)
            elif cfg == 'rate100':
                readout = STDPReadoutRate(1000, 8, seed=seed + 100, rate_window=100)
            elif cfg == 'rate50':
                readout = STDPReadoutRate(1000, 8, seed=seed + 100, rate_window=50)

            train_stdp(reservoir, readout, vis_enc, vt, targets, [0, 1, 2], 9000, 3.0)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0, 1, 2])
            avg = np.mean([results[s] for s in ['square', 'circle', 'triangle']])
            corrs.append(avg)
            _out(f"  {name:25s} seed {seed}: {avg:.3f}")
        mean_v = np.mean(corrs)
        std_v = np.std(corrs)
        readout_results[name] = corrs
        elapsed = time.time() - t_cfg
        _out(f"  >>> {name}: {mean_v:.3f} +/- {std_v:.3f} ({elapsed:.0f}s)")

    # === Statistics ===
    mem = np.array(readout_results['Membrane (original)'])
    rate = np.array(readout_results['Spike-rate w=100'])
    if scipy_stats is not None:
        t_stat, p_val = scipy_stats.ttest_ind(mem, rate)
    else:
        t_stat, p_val = 0.0, 1.0
    d = 0.0
    if np.std(mem) + np.std(rate) > 0:
        d = (np.mean(mem) - np.mean(rate)) / np.sqrt((np.std(mem)**2 + np.std(rate)**2) / 2)
    _out(f"\n  Membrane vs Rate-100: t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")

    # === Final Summary ===
    total = time.time() - t0
    _out("\n" + "=" * 60)
    _out("  FINAL SUMMARY")
    _out(f"  Total time: {total:.0f}s ({total/60:.1f} min)")
    _out("=" * 60)
    _out("\n  [A] RLS PSP Sweep:")
    for tau, corrs in rls_results.items():
        _out(f"    tau={tau:3d}ms: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")
    _out("\n  [B] Readout Comparison:")
    for name, corrs in readout_results.items():
        _out(f"    {name:25s}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")
    _out("\n  === RUN COMPLETE ===")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        _f.write("\nERROR:\n" + err + "\n")
        _f.flush()
        try:
            print("\nERROR:", flush=True)
            print(err, flush=True)
        except Exception:
            pass
    finally:
        _f.write("\n=== SCRIPT ENDED ===\n")
        _f.close()
