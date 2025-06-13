import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

class Logger:
    def __init__(self, smoothing_window=1000):  # Window in timesteps now
        self.smoothing_window = smoothing_window
        self.training_smoothing_window = smoothing_window // 10

        # Sample-based data
        self.timesteps = []
        self.rewards = []  # Individual step rewards
        self.values = []   # Value function estimates
        self.advantages = []   # Advantage estimates
        
        # Training data (logged per training update)
        self.training_steps = []   # When training occurred
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.log_stds = []  # Store log_std means
        
        # Rolling statistics
        self.reward_history = []  # Reward per timestep over windows
        self.advantage_estimate_history = []
        self.log_std_history = []

        self.policy_loss_history = [] # Over training steps
        self.value_loss_history = []
        self.entropy_history = []
        
        self.returns = []
        self.returns_history = []
        self.samples = []

    def log_step(self, timestep, reward, value=None, advantage=None):
        """Log individual timestep data"""
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if isinstance(advantage, torch.Tensor):
            advantage = advantage.cpu().numpy()

        self.timesteps.append(timestep)
        self.rewards.append(reward)
        
        if value is not None:
            self.values.append(value)
        if advantage is not None:
            self.advantages.append(advantage)
    
    def log_training_step(self, step, policy_loss, value_loss, entropy, log_std=None):
        """Log training losses at specific timesteps"""
        if isinstance(policy_loss, torch.Tensor):
            policy_loss = policy_loss.cpu().numpy()
        if isinstance(value_loss, torch.Tensor):
            value_loss = value_loss.cpu().numpy()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.cpu().numpy()
        if isinstance(log_std, torch.Tensor):
            log_std = log_std.cpu().numpy()

        self.training_steps.append(step)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        if log_std is not None:
            self.log_stds.append(log_std)
    
    def log_episode(self, episode_return, episode_length):
        """Optional: Log episode data if available"""
        if isinstance(episode_return, torch.Tensor):
            episode_return = episode_return.cpu().numpy()
        if isinstance(episode_length, torch.Tensor):
            episode_length = episode_length.cpu().numpy()
            
        self.returns.append(episode_return)
        if self.samples:
            self.samples.append(self.samples[-1] + episode_length)
        else:
            self.samples.append(episode_length)
    
    def _compute_rolling_average(self, data, window_size):
        """Helper to compute rolling average, padding the start."""
        if not data:
            return []
        
        series = pd.Series(data)
        # Use .rolling().mean() with min_periods to start from the first data point
        rolling_avg = series.rolling(window=window_size, min_periods=1).mean()
        return rolling_avg.tolist()

    def compute_rolling_metrics(self, window_size=None):
        """Compute rolling statistics over timestep windows"""
        if window_size is None:
            window_size = self.smoothing_window
            
        self.reward_rate_history = self._compute_rolling_average(self.rewards, window_size)
        self.value_estimate_history = self._compute_rolling_average(self.values, window_size)
        self.advantage_estimate_history = self._compute_rolling_average(self.advantages, window_size)
        
        loss_smoothing_window = max(10, self.training_smoothing_window) 
        self.policy_loss_history = self._compute_rolling_average(self.policy_losses, loss_smoothing_window)
        self.value_loss_history = self._compute_rolling_average(self.value_losses, loss_smoothing_window)
        self.entropy_history = self._compute_rolling_average(self.entropies, loss_smoothing_window)
        if self.log_stds:
            self.log_std_history = self._compute_rolling_average(self.log_stds, loss_smoothing_window)
        else:
            self.log_std_history = []

        # Ensure we have a valid window size for returns history (minimum of 1)
        returns_window = max(1, len(self.returns) // 10)
        self.returns_history = self._compute_rolling_average(self.returns, returns_window)

    def plot_training_curves(self, save_path=None):
        """Create comprehensive training plots for sample-based data"""
        # Compute rolling metrics first
        self.compute_rolling_metrics()

        fig, axes = plt.subplots(3, 3, figsize=(24, 18)) 
        fig.suptitle('RL Training Curves', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        plots = 0
        # 1. Rewards
        ax1 = axes[0]
        if self.rewards:
            ax1.plot(self.timesteps, self.rewards, alpha=0.2, color='lightblue', label='Raw Rewards')
            plots += 1
        if self.reward_rate_history:
            ax1.plot(self.timesteps[:len(self.reward_rate_history)], self.reward_rate_history, color='blue', linewidth=2, 
                     label=f'Rolling Avg ({self.smoothing_window} steps)')
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Reward')
            ax1.set_title('Reward vs Timesteps')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Advantages
        ax2 = axes[1]
        if self.advantages:
            ax2.plot(self.timesteps[:len(self.advantages)], self.advantages, alpha=0.2, color='lightcoral', label='Raw Advantages')
            plots += 1
        if self.advantage_estimate_history:
            ax2.plot(self.timesteps[:len(self.advantage_estimate_history)], self.advantage_estimate_history, color='red', linewidth=2, 
                     label=f'Rolling Avg ({self.smoothing_window} steps)')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Advantage Estimate')
            ax2.set_title('Advantages vs Timesteps')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Value Estimates
        ax3 = axes[2]
        if self.values:
            ax3.plot(self.timesteps, self.values, alpha=0.2, color='lightgreen', label='Raw Value Estimates')
            plots += 1
        if self.value_estimate_history:
            ax3.plot(self.timesteps[:len(self.value_estimate_history)], self.value_estimate_history, color='green', linewidth=2, 
                     label=f'Rolling Avg ({self.smoothing_window} steps)')
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Value Estimate')
            ax3.set_title('Value Estimates vs Timesteps')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Policy Loss
        ax4 = axes[3]
        if self.policy_losses:
            ax4.plot(self.training_steps, self.policy_losses, alpha=0.3, color='violet', label='Raw Policy Loss')
            plots += 1
        if self.policy_loss_history:
            ax4.plot(self.training_steps[:len(self.policy_loss_history)], self.policy_loss_history, color='purple', linewidth=2,
                     label=f'Rolling Avg ({max(10, len(self.policy_losses) // 10)} updates)') # Refer to actual window size
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Policy Loss')
            ax4.set_title('Policy Loss vs Training Steps')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Value Loss
        ax5 = axes[4]
        if self.value_losses:
            ax5.plot(self.training_steps, self.value_losses, alpha=0.3, color='wheat', label='Raw Value Loss')
            plots += 1
        if self.value_loss_history:
            # Ensure training_steps length matches history length for plotting from 0
            ax5.plot(self.training_steps[:len(self.value_loss_history)], self.value_loss_history, color='orange', linewidth=2,
                     label=f'Rolling Avg ({max(10, len(self.value_losses) // 10)} updates)') # Refer to actual window size
            ax5.set_xlabel('Timesteps')
            ax5.set_ylabel('Value Loss')
            ax5.set_title('Value Function Loss vs Timesteps')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Entropy
        ax6 = axes[5] 
        if self.entropies:
            ax6.plot(self.training_steps, self.entropies, alpha=0.3, color='lightsalmon', label='Raw Entropy')
            plots += 1
        if self.entropy_history:
            ax6.plot(self.training_steps[:len(self.entropy_history)], self.entropy_history, color='brown', linewidth=2,
                     label=f'Rolling Avg ({max(10, len(self.entropies) // 10)} updates)') # Refer to actual window size
            ax6.set_xlabel('Timesteps')
            ax6.set_ylabel('Entropy')
            ax6.set_title('Policy Entropy vs Timesteps')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. log_std
        if self.log_stds:
            ax7 = axes[6]
            ax7.plot(self.training_steps, self.log_stds, alpha=0.3, color='gray', label='Raw log_std')
            plots += 1
            if self.log_std_history:
                ax7.plot(self.training_steps[:len(self.log_std_history)], self.log_std_history, color='black', linewidth=2, 
                         label=f'Rolling Avg ({max(10, len(self.log_stds) // 10)} updates)')
                ax7.set_xlabel('Timesteps')
                ax7.set_ylabel('log_std')
                ax7.set_title('Policy log_std vs Timesteps')
                ax7.legend()
                ax7.grid(True, alpha=0.3)

        # Fill remaining subplot with a placeholder or remove it if not needed
        if len(axes) > plots:
            for i in range(plots, len(axes)):
                fig.delaxes(axes[i]) # Remove empty subplots
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Plot returns vs samples in a separate figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.samples, self.returns, alpha=0.3, color='lightblue', label='Raw Returns')
        if self.returns_history:
            ax.plot(self.samples[:len(self.returns_history)], self.returns_history, color='blue', linewidth=2, label='Rolling Returns')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Returns')
        ax.set_title('Returns vs Samples')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
# Example usage for sample-based logging
def example_sample_based_training():
    """Example of how to use sample-based logging"""
    logger = Logger(smoothing_window=1000)
    
    # Simulate training loop
    num_timesteps = 20000 
    
    for timestep in range(num_timesteps):
        # Simulate environment step
        reward = np.random.normal(0.1 + timestep * 0.00005, 0.5) if timestep > 5000 else np.random.normal(0.01, 0.1) 
        value = np.random.normal(timestep * 0.001, 0.5)  # Improving value estimates
        
        # Simulate advantage (e.g., oscillating around zero, with some positive spikes for good actions)
        advantage = np.random.normal(0, 1.0) 
        if timestep > num_timesteps / 2: # After some training, good actions might have positive advantages
            advantage = np.random.normal(0.5, 0.8) if np.random.rand() > 0.7 else np.random.normal(-0.2, 0.5)
        
        logger.log_step(timestep, reward, value, advantage)
        
        # Simulate training updates every 50 steps 
        if timestep % 50 == 0:
            policy_loss = np.random.exponential(1.0) * np.exp(-timestep/5000) + np.random.normal(0, 0.05) # Decreasing with noise
            value_loss = np.random.exponential(0.5) * np.exp(-timestep/3000) + np.random.normal(0, 0.03)
            entropy = np.random.normal(1.0, 0.1) * np.exp(-timestep/8000) + np.random.normal(0, 0.01) # Decreasing entropy
            
            logger.log_training_step(timestep, policy_loss, value_loss, entropy)

    # Simulate a few episodes ending for the episode plot
    for _ in range(1, 1000): # Simulate 1000 episodes
        ep_len = np.random.randint(1000, 2000) # Random episode length
        ep_return = np.random.uniform(50, 200) # Random episode return
        logger.log_episode(ep_return, ep_len)
    
    # Plot results
    logger.plot_training_curves()

if __name__ == "__main__":
    example_sample_based_training()
