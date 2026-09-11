"""
FORCE Phase 3: Local Plasticity (STDP + Eligibility Trace)
============================================================
Replace RLS with biologically plausible local rules.
Readout layer uses STDP + eligibility trace instead of global optimization.

Mechanisms (all from biology, all local):
1. STDP - pre before post = LTP, post before pre = LTD
2. Eligibility trace - synapse remembers recent activity,
   bridges the gap between motor command and sensory feedback
3. Homeostatic plasticity - neurons self-regulate firing rate

No RLS, no global error signal, no reward/punishment.

How to run:
    python force_phase3.py
"""
import numpy as np
import time
import os

# ============================================================
#  1. Spiking Reservoir (same as Phase 2, fixed weights)
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


# ============================================================
#  2. STDP Readout Layer with Eligibility Trace
# ============================================================
class STDPReadout:
    """
    Readout layer with local learning rules:
    - STDP: weight changes based on pre/post spike timing
    - Eligibility trace: bridges temporal gap between command and feedback
    - Homeostatic: output neurons self-regulate firing rate

    Key insight: During training, target force acts as "sensory feedback"
    that drives output neurons. STDP strengthens connections from reservoir
    neurons that are active when the output needs to fire.

    Parameters from biology:
    - STDP time window: 20ms (standard)
    - LTP amplitude: 0.01, LTD amplitude: 0.01 (balanced)
    - Eligibility trace tau: 100ms (bridges motor-sensory gap)
    - Homeostatic target rate: 10Hz
    """
    def __init__(self, n_reservoir, n_output=8, seed=42,
                 w_max=5.0, a_plus=0.01, a_minus=0.01,
                 tau_stdp=20.0, tau_elig=100.0,
                 homeo_target=10.0, homeo_gain=0.0001,
                 dt=1.0):
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

        self.rng = np.random.RandomState(seed)

        # Readout weights: start small, positive
        self.w_out = self.rng.rand(n_output, n_reservoir).astype(np.float32) * 0.1

        # STDP traces (pre-synaptic trace x, post-synaptic trace y)
        self.x_pre = np.zeros(n_reservoir, dtype=np.float32)
        self.y_post = np.zeros(n_output, dtype=np.float32)

        # Eligibility trace: per synapse, remembers recent co-activity
        self.elig = np.zeros((n_output, n_reservoir), dtype=np.float32)

        # Output neuron membrane potentials (LIF)
        self.v_out = np.zeros(n_output, dtype=np.float32)
        self.fired_out = np.zeros(n_output, dtype=np.float32)
        self.refractory_out = np.zeros(n_output, dtype=np.float32)

        # Homeostatic rate estimate
        self.rate_out = np.zeros(n_output, dtype=np.float32)

        # For monitoring
        self.output_rates = []

    def step(self, reservoir_fired, target_current=None):
        """
        Step the output layer.
        
        Args:
            reservoir_fired: binary spike vector from reservoir (n_in,)
            target_current: optional teaching current (n_out,)
                           If provided = training (guiding output)
                           If None = free running
        Returns:
            fired_out: output spikes
        """
        # Decay STDP traces
        decay_pre = np.exp(-self.dt / self.tau_stdp).astype(np.float32)
        decay_post = decay_pre
        self.x_pre = self.x_pre * decay_pre + reservoir_fired.astype(np.float32)
        self.y_post = self.y_post * decay_post + self.fired_out.astype(np.float32)

        # Decay eligibility trace
        decay_elig = np.exp(-self.dt / self.tau_elig).astype(np.float32)
        self.elig *= decay_elig

        # Update eligibility: when pre fires and post fires, eligibility increases
        # elig += outer(post_activity, pre_activity)
        co_activity = np.outer(self.fired_out.astype(np.float32),
                               reservoir_fired.astype(np.float32))
        self.elig += co_activity

        # Refractory countdown for output neurons
        self.refractory_out = np.maximum(0.0, self.refractory_out - self.dt)

        # Compute input current to output neurons
        I_syn = self.w_out @ reservoir_fired.astype(np.float32)

        # Add target current if training
        if target_current is not None:
            I_syn = I_syn + target_current

        # LIF dynamics for output neurons
        active = self.refractory_out <= 0.0
        tau_out = 10.0
        dv = (-self.v_out + I_syn) / tau_out * self.dt
        self.v_out[active] += dv[active]
        self.v_out[self.v_out < 0] = 0

        # Spike detection
        self.fired_out = (self.v_out >= 1.0).astype(np.float32)
        self.v_out[self.fired_out > 0.5] = 0.0
        self.refractory_out[self.fired_out > 0.5] = 2.0  # 2ms refractory

        # Homeostatic plasticity: adjust threshold based on firing rate
        self.rate_out = 0.999 * self.rate_out + 0.001 * self.fired_out
        rate_error = self.rate_out - self.homeo_target / 1000.0
        # If firing too much, decrease weights slightly; too little, increase
        homeo_adj = -self.homeo_gain * rate_error[:, np.newaxis]
        self.w_out += homeo_adj.astype(np.float32)

        # STDP update: only during training (target_current provided)
        if target_current is not None:
            # Guided STDP: target current drives output firing.
            # LTP: pre fired recently (high x_pre) + post fires now
            ltp = np.outer(self.fired_out.astype(np.float32), self.x_pre)
            # LTD: post fired recently (high y_post) + pre fires now
            ltd = np.outer(self.y_post, reservoir_fired.astype(np.float32))

            delta_w = (self.a_plus * ltp - self.a_minus * ltd).astype(np.float32)
            self.w_out += delta_w

        # Clip weights
        np.clip(self.w_out, 0.0, self.w_max, out=self.w_out)

        return self.fired_out.copy()

    def get_output(self, reservoir_fired):
        """Get continuous output from readout.
        Use weighted sum with centering (subtract mean) to get signed output."""
        raw = self.w_out @ reservoir_fired.astype(np.float32)
        # Center around mean to get positive and negative output
        centered = raw - raw.mean()
        return centered

    def get_output_rates(self):
        return self.rate_out * 1000.0


# ============================================================
#  3. Visual Encoder (same as Phase 2)
# ============================================================
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


# ============================================================
#  4. Vocal Tract Model (reused)
# ============================================================
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


# ============================================================
#  5. Target trajectory generator (reused)
# ============================================================
def generate_target_trajectory(shape_id, duration_steps=1500):
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
#  6. Phase 3: STDP training and testing
# ============================================================
def train_and_test_phase3(seed=42, n_neurons=1000,
                          train_steps=3000, test_steps=500,
                          force_gain=3.0, teach_current=2.0,
                          verbose=True):
    """
    Train with STDP + eligibility trace, test free running.
    
    Training: visual input + target current drives output neurons
              STDP strengthens active synapses
    Testing: visual input only, STDP off, see if output matches target
    """
    t0 = time.time()
    shape_names = ['square', 'circle', 'triangle']

    # Create components
    vis_enc = VisualEncoder(seed=seed)
    reservoir = SpikingReservoir(n=n_neurons, n_input=vis_enc.n_vis, seed=seed)
    readout = STDPReadout(n_neurons, n_output=8, seed=seed+100)
    vt = VocalTract(n_params=8, resistance=0.5, seed=seed+200)

    # Generate targets
    targets = [generate_target_trajectory(s, 1000) for s in range(3)]

    # ============================================================
    #  Training: 3 rounds x 3 shapes (sequential blocks)
    # ============================================================
    n_rounds = 3
    steps_per_block = train_steps // (n_rounds * 3)

    if verbose:
        print(f"[Training] {n_rounds} rounds x 3 shapes x {steps_per_block} steps", flush=True)

    for rnd in range(n_rounds):
        for shape_id in range(3):
            target = targets[shape_id]
            vt.reset()

            for step in range(steps_per_block):
                target_idx = step % len(target)

                # Visual input
                vis_spikes = vis_enc.encode(shape_id, rate=50.0)

                # Step reservoir
                fired = reservoir.step(vis_spikes)

                # Compute target force
                target_force = compute_target_force(
                    vt.position, vt.velocity, target[target_idx]
                )

                # Convert target force to teaching current for output neurons
                # Positive force -> positive current, negative force -> negative
                teach_curr = target_force * teach_current

                # Step output with teaching current (training mode)
                readout.step(fired, target_current=teach_curr)

                # Get output: use weighted sum (firing rate proxy)
                output = readout.get_output(fired)

                # Mix teacher and student force
                total_progress = (rnd * 3 + shape_id) * steps_per_block + step
                mix = min(1.0, total_progress / (train_steps * 0.5))
                mixed_force = (1 - mix) * target_force + mix * output / force_gain
                vt.step(mixed_force)

    train_time = time.time() - t0
    if verbose:
        print(f"  Training done ({train_time:.1f}s), rate={reservoir.get_firing_rate_hz():.1f}Hz")
        w_stats = f"w_out: mean={readout.w_out.mean():.4f}, max={readout.w_out.max():.4f}, sum={readout.w_out.sum():.1f}"
        print(f"  {w_stats}")

    # ============================================================
    #  Testing: visual only, no teaching current, STDP off
    # ============================================================
    if verbose:
        print(f"[Testing] 3 shapes x {test_steps} steps", flush=True)

    results = {}
    for shape_id in range(3):
        vt.reset()
        test_outputs = []
        test_targets = []

        for step in range(test_steps):
            vis_spikes = vis_enc.encode(shape_id, rate=50.0)
            fired = reservoir.step(vis_spikes)

            # No teaching current - free running
            readout.step(fired, target_current=None)
            output = readout.get_output(fired)

            # Scale output to force
            force = output * force_gain / 10.0  # normalize
            np.clip(force, -20.0, 20.0, out=force)
            vt.step(force)

            test_outputs.append(vt.position.copy())
            test_targets.append(targets[shape_id][step % len(targets[shape_id])].copy())

        test_outputs = np.array(test_outputs)
        test_targets = np.array(test_targets)
        corr = pearson_corr(test_outputs, test_targets)

        results[shape_names[shape_id]] = {
            'test_corr': corr,
            'outputs': test_outputs,
            'targets': test_targets,
        }
        if verbose:
            print(f"  {shape_names[shape_id]}: test_corr={corr:.3f}")

    total_time = time.time() - t0
    results['total_time'] = total_time
    results['firing_rate_hz'] = reservoir.get_firing_rate_hz()

    return results


# ============================================================
#  7. Main
# ============================================================
def main():
    print("=" * 65)
    print("  FORCE Phase 3: Local Plasticity (STDP + Eligibility)")
    print("  No RLS, no global error signal")
    print("=" * 65)

    seeds = [2, 3, 5]
    all_results = {}
    total_start = time.time()

    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        results = train_and_test_phase3(
            seed=seed, n_neurons=1000,
            train_steps=9000, test_steps=500,
            force_gain=3.0, teach_current=3.0,
            verbose=True
        )
        all_results[seed] = results

    total_time = time.time() - total_start

    # ============================================================
    #  Summary
    # ============================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 65)
    print()
    print(f"{'Seed':<8} {'Square':<12} {'Circle':<12} {'Triangle':<12}")
    print("-" * 48)

    all_corrs = []
    for seed in seeds:
        r = all_results[seed]
        sq = r['square']['test_corr']
        ci = r['circle']['test_corr']
        tr = r['triangle']['test_corr']
        all_corrs.extend([sq, ci, tr])
        print(f"{seed:<8} {sq:<12.3f} {ci:<12.3f} {tr:<12.3f}")

    print()
    shape_names = ['square', 'circle', 'triangle']
    for sn in shape_names:
        corrs = [all_results[s][sn]['test_corr'] for s in seeds]
        print(f"  {sn}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}")

    overall = np.mean(all_corrs)
    print(f"\n  Overall: {overall:.3f}")
    if overall > 0.5:
        print("  -> SUCCESS: local rules learned the mapping!")
    elif overall > 0.2:
        print("  -> PARTIAL: some learning, needs more training or tuning")
    else:
        print("  -> LOW: local rules not sufficient yet")

    save_dir = 'data_force_phase3'
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'results.npz'),
             all_results=all_results, allow_pickle=True)
    print(f"\nResults saved to {save_dir}/results.npz")


if __name__ == '__main__':
    main()
