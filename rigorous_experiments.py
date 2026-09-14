"""
Rigorous Academic Experiments for EGO Phase 3
===============================================
Complete experiment suite with proper controls,
ablations, scaling, generalization, and statistics.

Experiments:
1. Baseline comparison (RLS vs STDP vs no-guidance vs random)
2. Ablation study (remove each mechanism)
3. Scaling analysis (100 to 5000 neurons)
4. Generalization (train 2 shapes, test 3rd)
5. Parameter sensitivity (learning rate, eligibility tau, teach current)
6. 10-seed full validation with statistics

How to run:
    python rigorous_experiments.py
"""
import numpy as np
import time
import os
import json
from scipy import stats as scipy_stats

# ============================================================
#  Core Components (reused from Phase 3)
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

    def get_firing_rate_hz(self):
        return self.rate_ema / self.n * 1000.0 / self.dt


class STDPReadout:
    """STDP + eligibility trace + homeostatic plasticity readout layer."""
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


class RLSReadout:
    """Standard FORCE RLS readout for baseline comparison.
    Includes PSP filtering for fair comparison on spiking networks."""
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


# ============================================================
#  Training functions
# ============================================================
def train_stdp(reservoir, readout, vis_enc, vt, targets, shapes,
               train_steps, teach_current, n_rounds=3):
    """Train with STDP + eligibility + homeostasis."""
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
    """Train with RLS (FORCE baseline)."""
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
    """Test without guidance. Returns dict of shape_name -> correlation."""
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
#  Experiment 1: Baseline Comparison
# ============================================================
def exp1_baselines(seeds=list(range(10))):
    """Compare: STDP (ours) vs RLS (FORCE) vs no-guidance vs random weights.

    WARNING / 注意
    --------------
    The RLS baseline computed HERE is the ORIGINAL, UNFAIR version: it does
    not PSP-filter the reservoir spikes. It scores ~0.38 with huge variance
    and does NOT represent what RLS can achieve on this task.

    The FAIR baselines used in the paper come from `rls_fixed.py`
    (PSP-filtered, tau=20ms):
        ridge  = 0.873 +/- 0.010
        RLS    = 0.875 +/- 0.009

    This mismatch is why running this file reproduces 0.382 while the paper
    reports 0.875. See reanalysis/REPORT.md (section 6) for details.

    此处计算的 RLS 基线是**早期不公平的版本**（未对储备池脉冲做 PSP 滤波），
    得分约 0.38 且方差极大，不代表 RLS 在本任务上的真实水平。
    论文采用的公平基线在 rls_fixed.py 中（PSP 滤波，0.873 / 0.875）。
    """
    print("\n" + "="*65)
    print("  Experiment 1: Baseline Comparison")
    print("="*65)

    configs = {
        'STDP (ours)': {'method': 'stdp', 'teach_current': 3.0},
        'RLS (FORCE)': {'method': 'rls'},
        'No guidance': {'method': 'stdp', 'teach_current': 0.0},
        'Random weights': {'method': 'random'},
    }

    all_results = {}
    for config_name, config in configs.items():
        corrs_per_seed = []
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed+200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]

            if config['method'] == 'random':
                readout = STDPReadout(1000, 8, seed=seed+100,
                                       use_stdp=False, use_eligibility=False, use_homeostasis=False)
                results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
            elif config['method'] == 'stdp':
                readout = STDPReadout(1000, 8, seed=seed+100)
                teach_curr = config.get('teach_current', 3.0)
                train_stdp(reservoir, readout, vis_enc, vt, targets, [0,1,2],
                          9000, teach_curr)
                results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
            elif config['method'] == 'rls':
                readout = RLSReadout(1000, 8, seed=seed+100)
                train_rls(reservoir, readout, vis_enc, vt, targets, [0,1,2], 9000)
                results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])

            avg_corr = np.mean([results[s] for s in ['square','circle','triangle']])
            corrs_per_seed.append(avg_corr)

        all_results[config_name] = corrs_per_seed
        mean_val = np.mean(corrs_per_seed)
        std_val = np.std(corrs_per_seed)
        print(f"  {config_name:20s}: {mean_val:.3f} +/- {std_val:.3f}")

    # Statistics: STDP vs RLS
    stdp_arr = np.array(all_results['STDP (ours)'])
    rls_arr = np.array(all_results['RLS (FORCE)'])
    if len(seeds) >= 3:
        t_stat, p_val = scipy_stats.ttest_ind(stdp_arr, rls_arr)
        d = (np.mean(stdp_arr) - np.mean(rls_arr)) / np.sqrt((np.std(stdp_arr)**2 + np.std(rls_arr)**2) / 2) if np.std(stdp_arr) + np.std(rls_arr) > 0 else 0
        print(f"\n  STDP vs RLS: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")

    return all_results


# ============================================================
#  Experiment 2: Ablation Study
# ============================================================
def exp2_ablation(seeds=list(range(10))):
    """Remove each mechanism and measure impact."""
    print("\n" + "="*65)
    print("  Experiment 2: Ablation Study")
    print("="*65)

    configs = {
        'Full (STDP+Elig+Homeo)': {'use_stdp': True, 'use_eligibility': True, 'use_homeostasis': True},
        'No eligibility trace':   {'use_stdp': True, 'use_eligibility': False, 'use_homeostasis': True},
        'No homeostasis':         {'use_stdp': True, 'use_eligibility': True, 'use_homeostasis': False},
        'No STDP (elig+homeo only)': {'use_stdp': False, 'use_eligibility': True, 'use_homeostasis': True},
        'All removed (random)':   {'use_stdp': False, 'use_eligibility': False, 'use_homeostasis': False},
    }

    all_results = {}
    for config_name, config in configs.items():
        corrs_per_seed = []
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed+200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]
            readout = STDPReadout(1000, 8, seed=seed+100,
                                   use_stdp=config['use_stdp'],
                                   use_eligibility=config['use_eligibility'],
                                   use_homeostasis=config['use_homeostasis'])
            train_stdp(reservoir, readout, vis_enc, vt, targets, [0,1,2], 9000, 3.0)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
            corrs_per_seed.append(np.mean([results[s] for s in ['square','circle','triangle']]))

        all_results[config_name] = corrs_per_seed
        print(f"  {config_name:30s}: {np.mean(corrs_per_seed):.3f} +/- {np.std(corrs_per_seed):.3f}")

    return all_results


# ============================================================
#  Experiment 3: Scaling Analysis
# ============================================================
def exp3_scaling(seeds=list(range(3))):
    """Test with different network sizes."""
    print("\n" + "="*65)
    print("  Experiment 3: Scaling Analysis")
    print("="*65)

    sizes = [100, 300, 500, 1000, 2000, 5000]
    all_results = {}

    for n in sizes:
        corrs_per_seed = []
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=n, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed+200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]
            readout = STDPReadout(n, 8, seed=seed+100)
            train_stdp(reservoir, readout, vis_enc, vt, targets, [0,1,2], 9000, 3.0)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
            corrs_per_seed.append(np.mean([results[s] for s in ['square','circle','triangle']]))

        all_results[n] = corrs_per_seed
        print(f"  n={n:5d}: {np.mean(corrs_per_seed):.3f} +/- {np.std(corrs_per_seed):.3f}")

    return all_results


# ============================================================
#  Experiment 4: Generalization
# ============================================================
def exp4_generalization(seeds=list(range(10))):
    """Train on 2 shapes, test on held-out 3rd."""
    print("\n" + "="*65)
    print("  Experiment 4: Generalization (train 2, test 3rd)")
    print("="*65)

    hold_out_configs = [
        ('hold triangle', [0, 1], 2),
        ('hold circle', [0, 2], 1),
        ('hold square', [1, 2], 0),
    ]

    all_results = {}
    for config_name, train_shapes, test_shape in hold_out_configs:
        corrs_trained = []
        corrs_heldout = []
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed+200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]
            readout = STDPReadout(1000, 8, seed=seed+100)
            train_stdp(reservoir, readout, vis_enc, vt, targets, train_shapes, 9000, 3.0)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, train_shapes + [test_shape])
            shape_names = ['square', 'circle', 'triangle']
            for s in train_shapes:
                corrs_trained.append(results[shape_names[s]])
            corrs_heldout.append(results[shape_names[test_shape]])

        all_results[config_name] = {
            'trained': corrs_trained,
            'heldout': corrs_heldout
        }
        print(f"  {config_name}: trained={np.mean(corrs_trained):.3f}, heldout={np.mean(corrs_heldout):.3f}")

    return all_results


# ============================================================
#  Experiment 5: Parameter Sensitivity
# ============================================================
def exp5_sensitivity(seeds=list(range(3))):
    """Sweep key parameters."""
    print("\n" + "="*65)
    print("  Experiment 5: Parameter Sensitivity")
    print("="*65)

    # 5a: STDP learning rate
    print("\n  [5a] STDP learning rate (a_plus=a_minus):")
    lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    lr_results = {}
    for lr in lr_values:
        corrs = []
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed+200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]
            readout = STDPReadout(1000, 8, seed=seed+100, a_plus=lr, a_minus=lr)
            train_stdp(reservoir, readout, vis_enc, vt, targets, [0,1,2], 9000, 3.0)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
            corrs.append(np.mean([results[s] for s in ['square','circle','triangle']]))
        lr_results[lr] = corrs
        print(f"    lr={lr:.3f}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    # 5b: Eligibility trace tau
    print("\n  [5b] Eligibility trace time constant:")
    tau_values = [10, 50, 100, 200, 500]
    tau_results = {}
    for tau in tau_values:
        corrs = []
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed+200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]
            readout = STDPReadout(1000, 8, seed=seed+100, tau_elig=float(tau))
            train_stdp(reservoir, readout, vis_enc, vt, targets, [0,1,2], 9000, 3.0)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
            corrs.append(np.mean([results[s] for s in ['square','circle','triangle']]))
        tau_results[tau] = corrs
        print(f"    tau_elig={tau:4d}ms: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    # 5c: Teaching current strength
    print("\n  [5c] Teaching current strength:")
    tc_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    tc_results = {}
    for tc in tc_values:
        corrs = []
        for seed in seeds:
            vis_enc = VisualEncoder(seed=seed)
            reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
            vt = VocalTract(seed=seed+200)
            targets = [generate_target_trajectory(s, 1000) for s in range(3)]
            readout = STDPReadout(1000, 8, seed=seed+100)
            train_stdp(reservoir, readout, vis_enc, vt, targets, [0,1,2], 9000, tc)
            results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
            corrs.append(np.mean([results[s] for s in ['square','circle','triangle']]))
        tc_results[tc] = corrs
        print(f"    teach_curr={tc:.1f}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    return {'lr': lr_results, 'tau': tau_results, 'tc': tc_results}


# ============================================================
#  Experiment 6: 10-seed Full Validation
# ============================================================
def exp6_full_validation(seeds=list(range(10))):
    """Full 10-seed validation with per-shape statistics."""
    print("\n" + "="*65)
    print("  Experiment 6: 10-Seed Full Validation")
    print("="*65)

    shape_names = ['square', 'circle', 'triangle']
    all_shape_results = {s: [] for s in shape_names}
    all_avg = []

    for seed in seeds:
        vis_enc = VisualEncoder(seed=seed)
        reservoir = SpikingReservoir(n=1000, n_input=vis_enc.n_vis, seed=seed)
        vt = VocalTract(seed=seed+200)
        targets = [generate_target_trajectory(s, 1000) for s in range(3)]
        readout = STDPReadout(1000, 8, seed=seed+100)
        train_stdp(reservoir, readout, vis_enc, vt, targets, [0,1,2], 9000, 3.0)
        results = test_free_run(reservoir, readout, vis_enc, vt, targets, [0,1,2])
        for s in shape_names:
            all_shape_results[s].append(results[s])
        avg = np.mean([results[s] for s in shape_names])
        all_avg.append(avg)
        print(f"  seed={seed}: avg={avg:.3f}  ({', '.join(f'{s}={results[s]:.3f}' for s in shape_names)})")

    print(f"\n  --- Summary ---")
    for s in shape_names:
        arr = np.array(all_shape_results[s])
        ci = scipy_stats.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=scipy_stats.sem(arr))
        print(f"  {s:10s}: {np.mean(arr):.3f} +/- {np.std(arr):.3f}  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

    avg_arr = np.array(all_avg)
    ci = scipy_stats.t.interval(0.95, len(avg_arr)-1, loc=np.mean(avg_arr), scale=scipy_stats.sem(avg_arr))
    print(f"  {'Overall':10s}: {np.mean(avg_arr):.3f} +/- {np.std(avg_arr):.3f}  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

    return {'per_shape': all_shape_results, 'overall': all_avg}


# ============================================================
#  Main
# ============================================================
def main():
    t0 = time.time()
    print("="*65)
    print("  EGO Phase 3: Rigorous Academic Experiments")
    print("  Local plasticity (STDP + Eligibility + Homeostasis)")
    print("  No reward, no punishment, no global error signal")
    print("="*65)

    # Run all experiments
    # Use fewer seeds for expensive experiments
    seeds_10 = list(range(10))
    seeds_3 = list(range(3))

    results = {}

    # Exp 1: Baselines (10 seeds)
    results['exp1_baselines'] = exp1_baselines(seeds_10)

    # Exp 2: Ablation (10 seeds)
    results['exp2_ablation'] = exp2_ablation(seeds_10)

    # Exp 3: Scaling (3 seeds, expensive)
    results['exp3_scaling'] = exp3_scaling(seeds_3)

    # Exp 4: Generalization (10 seeds)
    results['exp4_generalization'] = exp4_generalization(seeds_10)

    # Exp 5: Parameter sensitivity (3 seeds each)
    results['exp5_sensitivity'] = exp5_sensitivity(seeds_3)

    # Exp 6: Full validation (10 seeds)
    results['exp6_full_validation'] = exp6_full_validation(seeds_10)

    total_time = time.time() - t0

    # ============================================================
    #  Final Summary
    # ============================================================
    print("\n" + "="*65)
    print("  FINAL SUMMARY")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("="*65)

    print("\n  [Exp1] Baselines:")
    for name, corrs in results['exp1_baselines'].items():
        print(f"    {name:20s}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    print("\n  [Exp2] Ablation:")
    for name, corrs in results['exp2_ablation'].items():
        print(f"    {name:30s}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    print("\n  [Exp3] Scaling:")
    for n, corrs in results['exp3_scaling'].items():
        print(f"    n={n:5d}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    print("\n  [Exp4] Generalization:")
    for name, data in results['exp4_generalization'].items():
        print(f"    {name}: trained={np.mean(data['trained']):.3f}, heldout={np.mean(data['heldout']):.3f}")

    print("\n  [Exp5] Parameter Sensitivity:")
    for param, sweeps in results['exp5_sensitivity'].items():
        print(f"    {param}:")
        for val, corrs in sweeps.items():
            print(f"      {val}: {np.mean(corrs):.3f}")

    print("\n  [Exp6] 10-Seed Validation:")
    val_data = results['exp6_full_validation']
    for s in ['square', 'circle', 'triangle']:
        arr = np.array(val_data['per_shape'][s])
        print(f"    {s:10s}: {np.mean(arr):.3f} +/- {np.std(arr):.3f}")
    avg_arr = np.array(val_data['overall'])
    print(f"    {'Overall':10s}: {np.mean(avg_arr):.3f} +/- {np.std(avg_arr):.3f}")

    # Save all results
    save_dir = 'data_rigorous'
    os.makedirs(save_dir, exist_ok=True)

    # Convert to JSON-serializable
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(os.path.join(save_dir, 'all_results.json'), 'w') as f:
        json.dump(make_json_safe(results), f, indent=2)

    np.savez(os.path.join(save_dir, 'all_results.npz'),
             results=make_json_safe(results), allow_pickle=True)

    print(f"\n  All results saved to {save_dir}/")
    print(f"\n  Done. Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
