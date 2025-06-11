import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

class Logger:
    def __init__(self, smoothing_window=1000):  # Window in timesteps now
        self.smoothing_window = smoothing_window
        
        # Sample-based data
        self.timesteps = []
        self.rewards = []  # Individual step rewards
        self.values = []   # Value function estimates
        self.advantages = []   # Advantage estimates
        
        # Loss data (logged per training update)
        self.training_timesteps = []   # When training occurred
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.log_stds = []  # Store log_std means
        
        # Rolling statistics
        self.cumulative_reward = 0
        self.reward_rate_history = []  # Reward per timestep over windows
        self.value_estimate_history = []
        self.advantage_estimate_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        if self.log_stds:
            self.log_std_history = []
        
        # Optional episode data (if available)
        self.episodes = []
        self.episode_returns = []
        self.episode_lengths = []
        
    def log_step(self, timestep, reward, value=None, advantage=None):
        """Log individual timestep data"""
        self.timesteps.append(timestep)
        self.rewards.append(reward)
        self.cumulative_reward += reward
        
        if value is not None:
            self.values.append(value)
        if advantage is not None:
            self.advantages.append(advantage)
    
    def log_training_step(self, timestep, policy_loss, value_loss, entropy, log_std=None):
        """Log training losses at specific timesteps"""
        self.training_timesteps.append(timestep)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        if log_std is not None:
            self.log_stds.append(log_std)
    
    def log_episode(self, episode_num, timestep, episode_return, episode_length):
        """Optional: Log episode data if available"""
        self.episodes.append(episode_num)
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
    
    def _compute_rolling_average(self, data, window_size):
        """Helper to compute rolling average, padding the start."""
        if not data:
            return []
        
        # Convert any CUDA tensors to CPU and then to numpy
        if isinstance(data[0], torch.Tensor):
            data = [d.cpu().numpy() if d.is_cuda else d.numpy() for d in data]
        
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
        
        # Here we'll use a dynamic window size for losses/entropy based on the number of actual training steps,
        # to ensure the smoothing is meaningful even if training steps are less frequent than env steps.
        # Let's say, 10% of total recorded training steps, or a minimum of 10.
        loss_smoothing_window = max(10, len(self.policy_losses) // 10) 

        self.policy_loss_history = self._compute_rolling_average(self.policy_losses, loss_smoothing_window)
        self.value_loss_history = self._compute_rolling_average(self.value_losses, loss_smoothing_window)
        self.entropy_history = self._compute_rolling_average(self.entropies, loss_smoothing_window)
        if self.log_stds:
            self.log_std_history = self._compute_rolling_average(self.log_stds, loss_smoothing_window)
        else:
            self.log_std_history = []

    def plot_training_curves(self, save_path=None):
        """Create comprehensive training plots for sample-based data"""
        # Compute rolling metrics first
        self.compute_rolling_metrics()

        fig, axes = plt.subplots(3, 3, figsize=(24, 18)) 
        fig.suptitle('RL Training Curves (Sample-Based)', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()

        # 1. Cumulative Reward vs Timesteps
        ax1 = axes[0]
        # Convert any CUDA tensors to CPU before cumsum
        rewards_cpu = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in self.rewards]
        cumulative_rewards = np.cumsum(rewards_cpu) if rewards_cpu else []
        ax1.plot(self.timesteps, cumulative_rewards, color='blue', linewidth=2)
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Cumulative Reward vs Timesteps')
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Average Reward Rate
        ax2 = axes[1]
        if self.rewards:
            ax2.plot(self.timesteps, rewards_cpu, alpha=0.2, color='lightblue', label='Raw Rewards')
        if self.reward_rate_history:
            ax2.plot(self.timesteps[:len(self.reward_rate_history)], self.reward_rate_history, color='blue', linewidth=2, 
                     label=f'Rolling Avg ({self.smoothing_window} steps)')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Reward per Step')
            ax2.set_title('Reward Rate vs Timesteps')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Value Function Estimates
        ax3 = axes[2]
        if self.values:
            values_cpu = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in self.values]
            ax3.plot(self.timesteps[:len(self.values)], values_cpu, alpha=0.3, color='lightgreen', label='Raw Values')
        if self.value_estimate_history:
            ax3.plot(self.timesteps[:len(self.value_estimate_history)], self.value_estimate_history, color='green', linewidth=2, 
                                 label=f'Rolling Avg ({self.smoothing_window} steps)')
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Value Estimate')
            ax3.set_title('Value Function Estimates')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Rolling Average Advantages
        ax4 = axes[3]
        if self.advantages:
            advantages_cpu = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in self.advantages]
            ax4.plot(self.timesteps[:len(self.advantages)], advantages_cpu, alpha=0.2, color='lightcoral', label='Raw Advantages')
        if self.advantage_estimate_history:
            ax4.plot(self.timesteps[:len(self.advantage_estimate_history)], self.advantage_estimate_history, color='red', linewidth=2, 
                     label=f'Rolling Avg ({self.smoothing_window} steps)')
            ax4.set_xlabel('Timesteps')
            ax4.set_ylabel('Advantage Estimate')
            ax4.set_title('Advantages vs Timesteps')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Policy Loss
        ax5 = axes[4]
        if self.policy_losses:
            ax5.plot(self.training_timesteps, self.policy_losses, alpha=0.3, color='violet', label='Raw Policy Loss')
        if self.policy_loss_history:
            ax5.plot(self.training_timesteps[:len(self.policy_loss_history)], self.policy_loss_history, color='purple', linewidth=2,
                     label=f'Rolling Avg ({max(10, len(self.policy_losses) // 10)} updates)') # Refer to actual window size
            ax5.set_xlabel('Timesteps')
            ax5.set_ylabel('Policy Loss')
            ax5.set_title('Policy Loss vs Timesteps')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Value Loss
        ax6 = axes[5]
        if self.value_losses:
            ax6.plot(self.training_timesteps, self.value_losses, alpha=0.3, color='wheat', label='Raw Value Loss')
        if self.value_loss_history:
            # Ensure training_timesteps length matches history length for plotting from 0
            ax6.plot(self.training_timesteps[:len(self.value_loss_history)], self.value_loss_history, color='orange', linewidth=2,
                     label=f'Rolling Avg ({max(10, len(self.value_losses) // 10)} updates)') # Refer to actual window size
            ax6.set_xlabel('Timesteps')
            ax6.set_ylabel('Value Loss')
            ax6.set_title('Value Function Loss vs Timesteps')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. Entropy
        ax7 = axes[6] 
        if self.entropies:
            ax7.plot(self.training_timesteps, self.entropies, alpha=0.3, color='lightsalmon', label='Raw Entropy')
        if self.entropy_history:
            ax7.plot(self.training_timesteps[:len(self.entropy_history)], self.entropy_history, color='brown', linewidth=2,
                     label=f'Rolling Avg ({max(10, len(self.entropies) // 10)} updates)') # Refer to actual window size
            ax7.set_xlabel('Timesteps')
            ax7.set_ylabel('Entropy')
            ax7.set_title('Policy Entropy vs Timesteps')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. log_std
        if hasattr(self, 'log_stds') and self.log_stds:
            ax8 = axes[7] if len(axes) > 7 else None
            if ax8:
                ax8.plot(self.training_timesteps, self.log_stds, alpha=0.3, color='gray', label='Raw log_std')
                if hasattr(self, 'log_std_history') and self.log_std_history:
                    ax8.plot(self.training_timesteps[:len(self.log_std_history)], self.log_std_history, color='black', linewidth=2,
                             label=f'Rolling Avg ({max(10, len(self.log_stds) // 10)} updates)')
                ax8.set_xlabel('Timesteps')
                ax8.set_ylabel('log_std')
                ax8.set_title('Policy log_std vs Timesteps')
                ax8.legend()
                ax8.grid(True, alpha=0.3)

        # Fill remaining subplot with a placeholder or remove it if not needed
        if len(axes) > 8:
            for i in range(8, len(axes)):
                fig.delaxes(axes[i]) # Remove empty subplots

        # If episode data is available, add episode markers to cumulative reward plot
        if self.episode_returns:
            episode_timesteps = []
            episode_cumulative_rewards = []
            cumulative = 0
            # Need to get the actual timestep where episodes ended based on logged data
            # Assuming episode_lengths are logged correctly to derive the timestep
            current_total_timesteps = 0
            for i, (ep_len, ep_ret) in enumerate(zip(self.episode_lengths, self.episode_returns)):
                current_total_timesteps += ep_len
                episode_timesteps.append(current_total_timesteps)
                episode_cumulative_rewards.append(cumulative_rewards[current_total_timesteps-1] if current_total_timesteps > 0 else 0)
            
            ax1.scatter(episode_timesteps, episode_cumulative_rewards, 
                                 color='red', alpha=0.6, s=20, label='Episode Ends')
            ax1.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_efficiency(self):
        """Plot sample efficiency metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reward accumulation rate
        ax1 = axes[0]
        if len(self.timesteps) > 1: # Ensure at least two timesteps for a meaningful rate
            # Calculate cumulative rewards
            rewards_cpu = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in self.rewards]
            cumulative_rewards_filtered = np.cumsum(rewards_cpu)
            timesteps_filtered = np.array(self.timesteps)
            
            # Find the first non-zero timestep index (or index > 0)
            start_idx = next((i for i, x in enumerate(timesteps_filtered) if x > 0), 1) 
            
            # Slice the data to avoid the initial problematic division
            plot_timesteps = timesteps_filtered[start_idx:]
            plot_cumulative_rewards = cumulative_rewards_filtered[start_idx:]
            
            # Calculate reward rate for plotting
            reward_rate = plot_cumulative_rewards / plot_timesteps
            
            ax1.plot(plot_timesteps, reward_rate, color='blue', linewidth=2)
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Cumulative Reward / Timesteps')
            ax1.set_title('Sample Efficiency (Reward Rate)')
            ax1.grid(True, alpha=0.3)
        else: # Handle case with very little data
            ax1.set_title('Sample Efficiency (Reward Rate)\n(Not enough data)')

        # Learning progress (if we have value estimates)
        ax2 = axes[1]
        if self.values and len(self.values) > 100:
            # Compare early vs late value estimates
            # Use rolling mean for more robust comparison if data is noisy
            avg_values_history = self._compute_rolling_average(self.values, self.smoothing_window)
            
            if len(avg_values_history) > 100:
                early_values = np.mean(avg_values_history[:len(avg_values_history)//4])
                late_values = np.mean(avg_values_history[-len(avg_values_history)//4:])
            else: # Fallback if not enough smoothed data
                early_values = np.mean(self.values[:len(self.values)//4])
                late_values = np.mean(self.values[-len(self.values)//4:])
            
            ax2.bar(['Early Training', 'Late Training'], [early_values, late_values], 
                             color=['lightcoral', 'lightgreen'])
            ax2.set_ylabel('Average Value Estimate')
            ax2.set_title('Learning Progress (Average Value)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_title('Learning Progress (Average Value)\n(Not enough value data)')
        
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
    current_timestep_for_episodes = 0
    for ep in range(1, 11): # Simulate 10 episodes
        ep_len = np.random.randint(1000, 2000) # Random episode length
        ep_return = np.random.uniform(50, 200) # Random episode return
        # Ensure the episode ends within the logged timesteps
        if current_timestep_for_episodes + ep_len < num_timesteps:
            current_timestep_for_episodes += ep_len
            logger.log_episode(ep, current_timestep_for_episodes, ep_return, ep_len)
    
    # Plot results
    logger.plot_training_curves()
    logger.plot_sample_efficiency()

if __name__ == "__main__":
    example_sample_based_training()
