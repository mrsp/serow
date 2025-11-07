import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from env_sac import SerowEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import shutil

def make_env_from_npz(path_npz, target_cf, w_pos=1.0, w_ori=0.5):
    data = np.load(path_npz, allow_pickle=True)
    return SerowEnv(data, target_cf, w_pos=w_pos, w_ori=w_ori)

# -------- main ----------
train_dataset = "datasets/go2_train.npz"
base_log = "logs_serow"
if os.path.exists(base_log):
    shutil.rmtree(base_log)

cfs = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
models = {}
reward_histories = {}

for cf in cfs:
    print(f"\nTraining {cf}...")
    
    env_vec = DummyVecEnv([lambda cf=cf: make_env_from_npz(train_dataset, cf)])
    env_vec = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_obs=20.0, gamma=0.995)
    
    model = SAC(
        "MlpPolicy",
        env_vec,
        verbose=1,
        seed = 42,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_starts=1_000,
        train_freq=(64, "step"),
        gradient_steps=64,
        ent_coef="auto_0.1",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                qf=[128, 128]
            )
        ),
    )
    
    model.learn(total_timesteps=20_000)
    model.save(f"serow_sac_{cf}")
    env_vec.save(f"vecnormalize_{cf}.pkl")
    models[cf] = model
    
    # Extract reward history from the environment
    reward_histories[cf] = env_vec.envs[0].reward_history.copy()

# Plot reward histories
plt.ioff()
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.ravel()

for ax, cf in zip(axes, cfs):
    rewards = np.array(reward_histories[cf])
    timesteps = np.arange(len(rewards))
    
    # Plot raw rewards
    ax.plot(timesteps, rewards, alpha=0.4, linewidth=0.5, label="step reward")
    
    # Plot moving average
    window = 100
    if len(rewards) >= window:
        rewards_ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        timesteps_ma = timesteps[window-1:]
        ax.plot(timesteps_ma, rewards_ma, linewidth=2, label=f"moving avg ({window})")
    
    ax.set_title(cf)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, linewidth=1)

fig.suptitle("Training Rewards per Contact Frame (Step-wise)")
fig.supxlabel("Timesteps")
fig.supylabel("Reward")
fig.tight_layout()
plt.show()
