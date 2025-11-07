import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from env_sac import SerowEnv, action_to_scale
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scipy.spatial.transform import Rotation as R
import serow

class TestSerowEnvMultiCF:
    """
    Test environment that uses separate RL models for each contact frame
    """
    def __init__(self, dataset):
        print("Initializing Test Serow Environment with Multi-CF support")
        
        # Load data from npz
        self.raw_imu_data = dataset["imu"]
        self.raw_joint_data = dataset["joints"]
        self.raw_ft_data = dataset["ft"]
        self.base_states = dataset["base_states"]
        self.joint_states = dataset["joint_states"] 
        self.pose_gt = dataset["base_pose_ground_truth"]
        self.contact_status = dataset["contact_states"]
        self.contact_frames = list(self.contact_status[0].contacts_status.keys())
        
        # Extract raw GT data
        self.pos_gt = []
        self.quat_gt = []
        for i in range(len(self.pose_gt)):
            self.pos_gt.append(np.array(self.pose_gt[i].position))
            self.quat_gt.append(np.array(self.pose_gt[i].orientation))

        self.pos_gt = np.array(self.pos_gt)
        self.quat_gt = np.array(self.quat_gt)
        
        # Normalize GT to start from origin
        self._normalize_ground_truth()
        
        self.N = len(self.pose_gt)
        
        print(f"Dataset size: {self.N} samples")
        print(f"Contact frames: {self.contact_frames}")
        print(f"Initial GT pose: pos={self.pos_gt[0]}, quat={self.quat_gt[0]}")
        
        # Storage for results
        self.baseline_orientations = []
        self.baseline_positions = []
        self.baseline_pos_errors = []
        self.baseline_ori_errors = []
        
        self.rl_positions = []
        self.rl_orientations = []
        self.rl_position_errors = []
        self.rl_ori_errors = []
        self.rl_rewards = []
        
        # Actions per contact frame
        self.rl_actions = {cf: [] for cf in self.contact_frames}
        
        # Action history per contact frame
        self.action_history_size = 10
        self.state_dim = 9 + self.action_history_size 
        self.action_history_buffers = {
            cf: np.zeros((self.action_history_size,), dtype=np.float32) 
            for cf in self.contact_frames
        }

    def _normalize_ground_truth(self):
        """Normalize ground truth to start from (0,0,0) and identity quaternion (1,0,0,0)"""
        # Get initial pose
        pos_0 = self.pos_gt[0].copy()
        quat_0 = self.quat_gt[0].copy()  # [w, x, y, z]
        
        # Create rotation from initial quaternion
        # scipy uses [x, y, z, w] convention
        R_0 = R.from_quat([quat_0[1], quat_0[2], quat_0[3], quat_0[0]])
        R_0_inv = R_0.inv()
        
        # Transform all poses
        normalized_pos = []
        normalized_quat = []
        
        for i in range(len(self.pos_gt)):
            # Transform position: rotate to initial frame, then subtract initial position
            pos_rotated = R_0_inv.apply(self.pos_gt[i] - pos_0)
            normalized_pos.append(pos_rotated)
            
            # Transform orientation: q_new = q_0_inv * q_i
            R_i = R.from_quat([self.quat_gt[i][1], self.quat_gt[i][2], 
                              self.quat_gt[i][3], self.quat_gt[i][0]])
            R_new = R_0_inv * R_i
            quat_new = R_new.as_quat()  # [x, y, z, w]
            # Convert back to [w, x, y, z]
            normalized_quat.append([quat_new[3], quat_new[0], quat_new[1], quat_new[2]])
        
        self.pos_gt = np.array(normalized_pos)
        self.quat_gt = np.array(normalized_quat)

    def compute_reward_and_error(self, state, step_idx):
        """Compute reward and errors"""
        pos_est = np.asarray(state.get_base_position(), dtype=np.float64)
        quat_gt = self.quat_gt[step_idx]
        quat_est = np.asarray(state.get_base_orientation(), dtype=np.float64)
        pos_gt = self.pos_gt[step_idx]
        quat_gt = self.quat_gt[step_idx]

        # Position error
        pos_err = float(np.linalg.norm(pos_est[0:2] - pos_gt[0:2]) + 
                       2 * np.linalg.norm(pos_est[2] - pos_gt[2]))
        
        # Orientation error
        quat_est_norm = quat_est / max(1e-12, np.linalg.norm(quat_est))
        quat_gt_norm = quat_gt / max(1e-12, np.linalg.norm(quat_gt))
        dot = np.clip(abs(np.dot(quat_est_norm, quat_gt_norm)), -1.0, 1.0)
        ori_err = 2.0 * np.arccos(dot)

        # Reward (w_pos=1.0, w_ori=0.5)
        reward = -(1.0 * pos_err + 0.5 * ori_err)

        return reward, pos_err, ori_err
    
    def get_observation(self, serow_framework, cf, state, kin, step_idx):
        """Generate observation for a specific contact frame"""
        obs = np.zeros((self.state_dim,), dtype=np.float32)
        
        # 1) Force of contact frame
        ft_t = self.raw_ft_data[step_idx]
        meas = ft_t.get(cf, None)
        if meas is None:
            obs[0:3] = 0.0
        else:
            force = np.asarray(meas.force, dtype=np.float32).ravel()
            if force.shape != (3,):
                force = force[:3]
            obs[0:3] = force

        # 2) Innovation and NIS
        success, innovation, S = serow_framework.get_contact_position_innovation(cf)

        if not success or innovation is None or S is None:
            obs[3:6] = 0.0
            obs[6] = 0.0
        else:
            v = np.asarray(innovation, dtype=np.float64).reshape(3,)
            S = np.asarray(S, dtype=np.float64).reshape(3, 3)
            S = 0.5 * (S + S.T)
            S.flat[::4] += 1e-9

            try:
                L = np.linalg.cholesky(S)
                y = np.linalg.solve(L, v)
                y = np.linalg.solve(L.T, y)
                nis = float(v @ y)
            except np.linalg.LinAlgError:
                y = np.linalg.solve(S, v)
                nis = float(v @ y)

            obs[3:6] = v.astype(np.float32)
            obs[6] = np.float32(nis)

        # 3) Frobenius norm of measurement covariance
        Rm = np.asarray(kin.contacts_position_noise[cf] + kin.position_cov, 
                       dtype=np.float64).reshape(3, 3)
        frob_R = np.linalg.norm(Rm, ord='fro')
        obs[7] = np.float32(frob_R)

        # 4) Action history for THIS contact frame
        n = self.action_history_size
        obs[8:8+n] = self.action_history_buffers[cf][-n:]

        return obs
    
    def run_baseline_test(self, dataset):
        """Baseline with all actions = 1.0"""
        print(f"\n=== Running BASELINE test (all CFs action=1.0) ===")
        serow_framework = serow.Serow()
        serow_framework.initialize("go2_rl.json")
        initial_state = serow_framework.get_state(allow_invalid=True)
        initial_state.set_base_state(dataset["base_states"][0])
        initial_state.set_joint_state(dataset["joint_states"][0])
        initial_state.set_contact_state(dataset["contact_states"][0])
        serow_framework.set_state(initial_state)
        
        self.baseline_pos_errors = []
        self.baseline_ori_errors = []
        
        idx = 0
        for imu, joint, ft in zip(dataset["imu"], dataset["joints"], dataset["ft"]):
            imu, kin, ft = serow_framework.process_measurements(imu, joint, ft, None)
            serow_framework.base_estimator_predict_step(imu, kin)
            
            # Apply baseline action to all contact frames
            for cf in kin.contacts_status.keys():
                serow_framework.set_action(cf, np.array([1.0], dtype=np.float32))
                serow_framework.base_estimator_update_with_contact_position(cf, kin)
            
            serow_framework.base_estimator_finish_update(imu, kin)
            state = serow_framework.get_state(allow_invalid=True)
            
            self.baseline_positions.append(state.get_base_position())
            self.baseline_orientations.append(state.get_base_orientation())
            
            _, pos_err, ori_err = self.compute_reward_and_error(state, idx)
            self.baseline_pos_errors.append(pos_err)
            self.baseline_ori_errors.append(ori_err)
            
            idx += 1
        
        self.baseline_positions = np.array(self.baseline_positions)
        self.baseline_orientations = np.array(self.baseline_orientations)
        self.baseline_pos_errors = np.array(self.baseline_pos_errors)
        self.baseline_ori_errors = np.array(self.baseline_ori_errors)
        
        print(f"Baseline Pos Error - Mean: {np.mean(self.baseline_pos_errors):.4f}, Std: {np.std(self.baseline_pos_errors):.4f}")
        print(f"Baseline Ori Error - Mean: {np.mean(self.baseline_ori_errors):.4f}, Std: {np.std(self.baseline_ori_errors):.4f}")

    def run_rl_agent_test(self, models, vec_normalizes):
        """
        Run test with trained RL agents for each contact frame
        
        Args:
            models: dict mapping contact_frame_name -> SAC model
            vec_normalizes: dict mapping contact_frame_name -> VecNormalize
        """
        serow_framework = serow.Serow()
        serow_framework.initialize("go2_rl.json")
        initial_state = serow_framework.get_state(allow_invalid=True)
        initial_state.set_base_state(self.base_states[0])
        initial_state.set_joint_state(self.joint_states[0])
        initial_state.set_contact_state(self.contact_status[0])
        serow_framework.set_state(initial_state)      
        
        print(f"\n=== Running RL AGENT test with all {len(models)} models ===")
        step_count = 0
        
        for imu, joint, ft in zip(self.raw_imu_data, self.raw_joint_data, self.raw_ft_data):
            imu, kin, ft = serow_framework.process_measurements(imu, joint, ft, None)
            serow_framework.base_estimator_predict_step(imu, kin)
            
            # Get state after prediction for observation generation
            post_state = serow_framework.get_state(allow_invalid=True)
            
            # Process each contact frame
            for cf in self.contact_frames:
                if kin.contacts_status[cf]:  # If in contact
                    # Get observation for this contact frame
                    obs = self.get_observation(serow_framework, cf, post_state, kin, step_count)
                    
                    # Normalize observation
                    if vec_normalizes[cf] is not None:
                        obs = vec_normalizes[cf].normalize_obs(obs)
                    
                    # Get action from RL model for this contact frame
                    action, _ = models[cf].predict(obs, deterministic=True)
                    action = np.array(action)
                    
                    # Convert to scale and apply
                    scale = action_to_scale(action)
                    serow_framework.set_action(cf, scale)
                    
                    # Store action
                    self.rl_actions[cf].append(scale[0])
                    
                    # Update action history for this CF
                    self.action_history_buffers[cf] = np.roll(self.action_history_buffers[cf], -1)
                    self.action_history_buffers[cf][-1] = action[0]
                else:
                    # Not in contact - use baseline or don't update
                    scale = np.array([1.0], dtype=np.float32)
                    serow_framework.set_action(cf, scale)
                
                # Apply update for this contact frame
                serow_framework.base_estimator_update_with_contact_position(cf, kin)
            
            # Finish update
            serow_framework.base_estimator_finish_update(imu, kin)
            
            # Get updated state
            state = serow_framework.get_state(allow_invalid=True)
            
            # Store results
            pos_est = np.array(state.get_base_position())
            quat_est = np.array(state.get_base_orientation())
            self.rl_positions.append(pos_est.copy())
            self.rl_orientations.append(quat_est.copy())
            
            # Compute reward and errors
            reward, pos_err, ori_err = self.compute_reward_and_error(state, step_count)
            self.rl_rewards.append(reward)
            self.rl_position_errors.append(pos_err)
            self.rl_ori_errors.append(ori_err)
            
            step_count += 1
            
            if step_count % 1000 == 0:
                print(f"RL Agent - Step {step_count}/{self.N-1}, Pos Error: {pos_err:.4f}")
        
        print("RL agent test completed!")
        
        # Convert to arrays
        self.rl_positions = np.array(self.rl_positions)
        self.rl_orientations = np.array(self.rl_orientations)
        self.rl_rewards = np.array(self.rl_rewards)
        self.rl_position_errors = np.array(self.rl_position_errors)
        self.rl_ori_errors = np.array(self.rl_ori_errors)
        
        # Convert actions to arrays
        for cf in self.contact_frames:
            self.rl_actions[cf] = np.array(self.rl_actions[cf])
            
            
def plot_comparison_results(test_env):
    """Create comparison plots with 4-column layout: position errors, quaternions, and cumulative metrics."""
    time_steps = np.arange(len(test_env.rl_positions))
    
    # 3 rows × 4 cols layout
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # --- Row 1: Position Errors (X, Y, Z) + Total Position Error ---
    # X Position Error
    x_error_rl = np.abs(test_env.rl_positions[:, 0] - test_env.pos_gt[:, 0])
    x_error_baseline = np.abs(test_env.baseline_positions[:, 0] - test_env.pos_gt[:, 0])
    axes[0, 0].plot(time_steps, test_env.pos_gt[:, 0], 'r-', label='Ground Truth', linewidth=1.5)
    axes[0, 0].plot(time_steps, test_env.rl_positions[:, 0], 'b-', label='RL Agent', linewidth=1, alpha=0.8)
    axes[0, 0].plot(time_steps, test_env.baseline_positions[:, 0], 'g--', label='Baseline', linewidth=1, alpha=0.8)
    axes[0, 0].set_ylabel('X Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Y Position Error
    y_error_rl = np.abs(test_env.rl_positions[:, 1] - test_env.pos_gt[:, 1])
    y_error_baseline = np.abs(test_env.baseline_positions[:, 1] - test_env.pos_gt[:, 1])
    axes[0, 1].plot(time_steps, test_env.pos_gt[:, 1], 'r-', label='Ground Truth', linewidth=1.5)
    axes[0, 1].plot(time_steps, test_env.rl_positions[:, 1], 'b-', label='RL Agent', linewidth=1, alpha=0.8)
    axes[0, 1].plot(time_steps, test_env.baseline_positions[:, 1], 'g--', label='Baseline', linewidth=1, alpha=0.8)
    axes[0, 1].set_ylabel('Y Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z Position Error
    z_error_rl = np.abs(test_env.rl_positions[:, 2] - test_env.pos_gt[:, 2])
    z_error_baseline = np.abs(test_env.baseline_positions[:, 2] - test_env.pos_gt[:, 2])
    axes[0, 2].plot(time_steps, test_env.pos_gt[:, 2], 'r-', label='Ground Truth', linewidth=1.5)
    axes[0, 2].plot(time_steps, test_env.rl_positions[:, 2], 'b-', label='RL Agent', linewidth=1, alpha=0.8)
    axes[0, 2].plot(time_steps, test_env.baseline_positions[:, 2], 'g--', label='Baseline', linewidth=1, alpha=0.8)
    axes[0, 2].set_ylabel('Z Position (m)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Total Position Error
    axes[0, 3].plot(time_steps, test_env.rl_position_errors, 'b-', label='RL Agent', linewidth=1)
    axes[0, 3].plot(time_steps, test_env.baseline_pos_errors, 'g--', label='Baseline', linewidth=1)
    axes[0, 3].set_ylabel('Total Position Error (m)')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    # --- Row 2: Quaternions (W, X, Y, Z) ---
    quat_labels = ['W', 'X', 'Y', 'Z']
    for i in range(4):
        axes[1, i].plot(time_steps, test_env.quat_gt[:, i], 'r-', label='Ground Truth', linewidth=1.5)
        axes[1, i].plot(time_steps, test_env.rl_orientations[:, i], 'b-', label='RL Agent', linewidth=1, alpha=0.8)
        # Assuming baseline orientations are stored similarly
        if hasattr(test_env, 'baseline_orientations'):
            axes[1, i].plot(time_steps, test_env.baseline_orientations[:, i], 'g--', label='Baseline', linewidth=1, alpha=0.8)
        axes[1, i].set_ylabel(f'Quaternion {quat_labels[i]}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    # --- Row 3: Cumulative Error, Orientation Error, XY Trajectory, and Cumulative Ori Error ---
    # Cumulative Position Error
    cumulative_rl = np.cumsum(test_env.rl_position_errors)
    cumulative_baseline = np.cumsum(test_env.baseline_pos_errors)
    axes[2, 0].plot(time_steps, cumulative_rl, 'b-', label='RL Agent', linewidth=1.5)
    axes[2, 0].plot(time_steps, cumulative_baseline, 'g--', label='Baseline', linewidth=1.5)
    axes[2, 0].set_xlabel('Time Steps')
    axes[2, 0].set_ylabel('Cumulative Error (m)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Orientation Error
    axes[2, 1].plot(time_steps, test_env.rl_ori_errors, 'b-', label='RL Agent', linewidth=1)
    axes[2, 1].plot(time_steps, test_env.baseline_ori_errors, 'g--', label='Baseline', linewidth=1)
    axes[2, 1].set_xlabel('Time Steps')
    axes[2, 1].set_ylabel('Orientation Error (rad)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # XY Trajectory
    axes[2, 2].plot(test_env.pos_gt[:, 0], test_env.pos_gt[:, 1], 'r-', label='Ground Truth', linewidth=1.5)
    axes[2, 2].plot(test_env.rl_positions[:, 0], test_env.rl_positions[:, 1], 'b-', label='RL Agent', linewidth=1, alpha=0.8)
    axes[2, 2].plot(test_env.baseline_positions[:, 0], test_env.baseline_positions[:, 1], 'g--', label='Baseline', linewidth=1, alpha=0.8)
    axes[2, 2].set_xlabel('X Position (m)')
    axes[2, 2].set_ylabel('Y Position (m)')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].axis('equal')
    
    # Cumulative Orientation Error
    cumulative_ori_rl = np.cumsum(test_env.rl_ori_errors)
    cumulative_ori_baseline = np.cumsum(test_env.baseline_ori_errors)
    axes[2, 3].plot(time_steps, cumulative_ori_rl, 'b-', label='RL Agent', linewidth=1.5)
    axes[2, 3].plot(time_steps, cumulative_ori_baseline, 'g--', label='Baseline', linewidth=1.5)
    axes[2, 3].set_xlabel('Time Steps')
    axes[2, 3].set_ylabel('Cumulative Ori Error (rad)')
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_vs_baseline_4col.png', dpi=150)
    print("Plot saved as 'rl_vs_baseline_4col.png'")
    plt.show()
    
def load_models_and_normalizers(dataset, contact_frames):
    """Load all trained models and their normalizers"""
    models = {}
    vec_normalizes = {}
    
    for cf in contact_frames:
        print(f"\nLoading model for {cf}...")
        try:
            # Load model
            model_path = f"serow_sac_{cf}"
            models[cf] = SAC.load(model_path)
            print(f"  ✓ Loaded model: {model_path}")
            
            # Load VecNormalize
            try:
                from env_sac import SerowEnv
                dummy_env = DummyVecEnv([lambda: SerowEnv(dataset, cf)])
                vec_norm_path = f"vecnormalize_{cf}.pkl"
                vec_normalize = VecNormalize.load(vec_norm_path, dummy_env)
                vec_normalize.training = False
                vec_normalize.norm_reward = False
                vec_normalizes[cf] = vec_normalize
                print(f"  ✓ Loaded normalizer: {vec_norm_path}")
            except Exception as e:
                print(f"  ⚠ Could not load normalizer for {cf}: {e}")
                vec_normalizes[cf] = None
                
        except Exception as e:
            print(f"  ✗ Failed to load model for {cf}: {e}")
            raise
    
    print(f"\n✓ Successfully loaded {len(models)} models")
    return models, vec_normalizes

def main():    
    # Load test dataset
    test_dataset = "go2_test_slope.npz"
    data = np.load(test_dataset, allow_pickle=True)
    print(f"Loaded dataset: {test_dataset}")
    
    # Create test environment
    test_env = TestSerowEnvMultiCF(data)
    
    # Run baseline test
    test_env.run_baseline_test(data)
    
    # Load all trained models
    models, vec_normalizes = load_models_and_normalizers(data, test_env.contact_frames)
    
    # Run RL agent test with all models
    test_env.run_rl_agent_test(models, vec_normalizes)
    
    # Plot comparison results
    plot_comparison_results(test_env)

if __name__ == "__main__":
    main()
