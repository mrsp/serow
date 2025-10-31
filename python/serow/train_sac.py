import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from env_sac import SerowEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import shutil



def make_env_from_npz(path_npz, target_cf, log_dir, w_pos=1.0, w_ori=0.5):
    os.makedirs(log_dir, exist_ok=True)
    data = np.load(path_npz, allow_pickle=True)
    return Monitor(SerowEnv(data, target_cf, w_pos=w_pos, w_ori=w_ori),
                   filename=os.path.join(log_dir, "monitor.csv"))

def get_xy_from_monitor(log_dir):
    df = load_results(log_dir)
    x, y = ts2xy(df, "timesteps")  # y = ep rewards
    return np.asarray(x), np.asarray(y)

def moving_avg(y, k=20):
    if len(y) < k: return y, np.arange(len(y))
    y_ma = np.convolve(y, np.ones(k)/k, mode="valid")
    x_ma = np.arange(len(y_ma))
    return y_ma, x_ma

# -------- main ----------
train_dataset = "datasets/go2_train.npz"
base_log = "logs_serow"
if os.path.exists(base_log):
    shutil.rmtree(base_log)
os.makedirs(base_log, exist_ok=True)

cfs = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
log_dirs = {}
models = {}

for cf in cfs:
    print(f"\nTraining {cf}...")
    log_dir = os.path.join(base_log, cf)
    log_dirs[cf] = log_dir

    env_vec = DummyVecEnv([lambda cf=cf, ld=log_dir: make_env_from_npz(train_dataset, cf, ld)])
    env_vec = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    model = SAC(
        "MlpPolicy",
        env_vec,
        verbose=1,
        learning_rate=1e-5,
        buffer_size=50_000,
        batch_size=256,
        gamma=0.995,
        tau=0.005,
        learning_starts=5_000,
        train_freq=(64, "step"),
        gradient_steps=128,
        ent_coef="auto_0.1",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
    )

    model.learn(total_timesteps=75_000)
    model.save(f"serow_sac_{cf}")
    env_vec.save(f"vecnormalize_{cf}.pkl")
    models[cf] = model

plt.ioff()  # ensure non-interactive during script runs

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.ravel()

for ax, cf in zip(axes, cfs):
    x, y = get_xy_from_monitor(log_dirs[cf])
    y_ma, x_ma = moving_avg(y, k=20)

    ax.plot(x, y, alpha=0.35, label="raw ep reward")
    # Align MA x with timesteps (use the timesteps that correspond to the kept episodes)
    if len(y_ma) > 0:
        ax.plot(x[len(x)-len(y_ma):], y_ma, linewidth=2, label="moving avg (20)")
    ax.set_title(cf)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Training rewards per contact frame (raw & moving average)")
fig.supxlabel("Timesteps")
fig.supylabel("Episode reward (raw)")
fig.tight_layout()
plt.show()
