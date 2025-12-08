import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from env_sac import SerowEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import shutil

def make_env_from_npz(path_npz, target_cf):
    data = np.load(path_npz, allow_pickle=True)
    return SerowEnv(data, target_cf)

# -------- main ----------
train_dataset = "datasets/train/go2_train_long.npz"
val_dataset = "datasets/test/go2_test_slope.npz"
base_log = "logs_serow"
if os.path.exists(base_log):
    shutil.rmtree(base_log)

cfs = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
models = {}
reward_histories = {}
episode_histories = {}

for cf in cfs:
    print(f"\nTraining {cf}...")
    
    eval_env = DummyVecEnv([lambda cf=cf: make_env_from_npz(val_dataset, cf)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        training=True  # Don't update running stats during eval
    )


    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-1.5, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        n_eval_episodes=50,
        eval_freq=5_000,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        deterministic=True,
        verbose=1
    )


    env_vec = DummyVecEnv([lambda cf=cf: make_env_from_npz(train_dataset, cf)])
    env_vec = VecNormalize(env_vec, norm_obs=True, norm_reward=False,clip_reward = 10.0, clip_obs=10.0, gamma=0.99)
    
    model = SAC(
        "MlpPolicy",
        env_vec,
        verbose=1,
        seed = 42,
        learning_rate=3e-4,
        buffer_size=20_000,
        batch_size=512,
        gamma=0.99,
        tau=0.005,
        learning_starts=5_000,
        train_freq=(1, "step"),
        gradient_steps=10,
        ent_coef="auto_0.1",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                qf=[128, 128]
            )
        ),
    )
    
    model.learn(total_timesteps=50_000, progress_bar=True)
    model.save(f"serow_sac_{cf}")
    env_vec.save(f"vecnormalize_{cf}.pkl")
    models[cf] = model
    
    # Extract reward history from the environment
    reward_histories[cf] = env_vec.envs[0].reward_history.copy()


for cf in cfs:
    print(f"\n{cf} Episode Statistics:")
    
    # Access the unwrapped environment
    env_unwrapped = env_vec.envs[0]  # Get first (and only) env from DummyVecEnv
    
    episode_returns = env_unwrapped.episode_returns
    episode_lengths = env_unwrapped.episode_lengths
    
    print(f"  Total episodes: {len(episode_returns)}")
    print(f"  Mean return: {np.mean(episode_returns):.2f}")
    print(f"  Std return: {np.std(episode_returns):.2f}")
    print(f"  Best return: {np.max(episode_returns):.2f}")
    print(f"  Worst return: {np.min(episode_returns):.2f}")
    
    # Store for plotting later
    models[cf] = model
    episode_histories[cf] = episode_returns
    
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, cf in enumerate(cfs):
    ax = axes[idx]
    returns = episode_histories[cf]
    episodes = np.arange(len(returns))
    
    ax.plot(episodes, returns, alpha=0.3, label='Episode return')
    
    # Moving average
    if len(returns) >= 20:
        ma = np.convolve(returns, np.ones(20)/20, mode='valid')
        ax.plot(np.arange(19, len(returns)), ma, linewidth=2, label='Moving avg (20)')
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(cf)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("episode_returns.png", dpi=150)
plt.show()


# # Plot reward histories
# plt.ioff()
# fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
# axes = axes.ravel()

# for ax, cf in zip(axes, cfs):
#     rewards = np.array(reward_histories[cf])
#     timesteps = np.arange(len(rewards))
    
#     # Plot raw rewards
#     ax.plot(timesteps, rewards, alpha=0.4, linewidth=0.5, label="step reward")
    
#     # Plot moving average
#     window = 100
#     if len(rewards) >= window:
#         rewards_ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
#         timesteps_ma = timesteps[window-1:]
#         ax.plot(timesteps_ma, rewards_ma, linewidth=2, label=f"moving avg ({window})")
    
#     ax.set_title(cf)
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, linewidth=1)

# fig.suptitle("Training Rewards per Contact Frame (Step-wise)")
# fig.supxlabel("Timesteps")
# fig.supylabel("Reward")
# fig.tight_layout()
# plt.show()
