import serow
import numpy as np

from utils import quaternion_to_rotation_matrix, logMap, sync_and_align_data, plot_trajectories

class RewardShaper:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.recent_nis = []
        self.recent_position_errors = []
        self.recent_orientation_errors = []

    def _update_buffer(self, buffer, value):
        buffer.append(value)
        if len(buffer) > self.buffer_size:
            buffer.pop(0)

    def _get_alpha(self, buffer, time_factor=1.0, percentile=95):
        if len(buffer) > 10:
            typical = np.percentile(buffer, percentile)
            if typical * time_factor > 0:
                return -np.log(0.5) / (typical * time_factor + 1e-8)
        return 1.0

    def compute_reward(self, cf, serow_framework, state, gt, step, use_ground_truth=True):
        success = False
        innovation = np.zeros(3)
        covariance = np.zeros((3, 3))
        success, _, _, innovation, covariance = serow_framework.get_contact_position_innovation(cf)
        reward = None
        done = None

        STEP_REWARD = 0.01
        DIVERGENCE_PENALTY = -5.0  
        TIME_SCALE = 0.05

        if success:
            done = 0.0
            try:
                reg_covariance = covariance + np.eye(3) * 1e-6
                nis = innovation.dot(np.linalg.inv(reg_covariance).dot(innovation))
            except np.linalg.LinAlgError:
                nis = float('inf')

            self._update_buffer(self.recent_nis, nis)
            if nis > 25.0 or nis <= 0.0: 
                reward = DIVERGENCE_PENALTY 
                done = 1.0  
            else:
                alpha_nis = self._get_alpha(self.recent_nis, percentile=90)
                innovation_reward = np.exp(-alpha_nis * nis)
                reward = innovation_reward + STEP_REWARD

                if use_ground_truth:
                    time_factor = 1.0 + TIME_SCALE * step

                    # Position error
                    position_error = np.linalg.norm(state.get_base_position() - gt.position)
                    self._update_buffer(self.recent_position_errors, position_error)
                    alpha_pos = self._get_alpha(self.recent_position_errors, time_factor, percentile=85)
                    position_reward = np.exp(-alpha_pos * position_error * time_factor)

                    # Orientation error
                    orientation_error = np.linalg.norm(logMap(quaternion_to_rotation_matrix(gt.orientation).transpose() 
                                                              @ quaternion_to_rotation_matrix(state.get_base_orientation())))
                    self._update_buffer(self.recent_orientation_errors, orientation_error)
                    alpha_ori = self._get_alpha(self.recent_orientation_errors, time_factor, percentile=70)
                    orientation_reward = np.exp(-alpha_ori * orientation_error * time_factor)
                    
                    reward += position_reward + orientation_reward

                reward /= abs(DIVERGENCE_PENALTY)
        return reward, done

class SerowEnv:
    def __init__(self, robot, joint_state, base_state, contact_state):
        self.robot = robot
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{robot}_rl.json")
        self.initial_state = self.serow_framework.get_state(allow_invalid=True)
        self.initial_state.set_joint_state(joint_state)
        self.initial_state.set_base_state(base_state)  
        self.initial_state.set_contact_state(contact_state)
        self.serow_framework.set_state(self.initial_state)
        self.contact_frames = self.initial_state.get_contacts_frame()
        self.action_dim = 1
        self.state_dim = 7
        self.reward_shaper = RewardShaper()

    def _compute_reward(self, cf, state, gt, step, use_ground_truth=True):
        return self.reward_shaper.compute_reward(cf, self.serow_framework, state, gt, step, 
                                                 use_ground_truth=use_ground_truth)

    def compute_state(self, cf, state, contact_state):
        R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
        local_pos = R_base @ (state.get_base_position() - state.get_contact_position(cf))
        local_pos = np.array([abs(local_pos[0]), abs(local_pos[1]), local_pos[2]])
        local_vel = R_base @ state.get_base_linear_velocity()  
        return np.concatenate((local_pos, local_vel, 
                               np.array([contact_state.contacts_probability[cf]])), axis=0)

    def reset(self):
        self.serow_framework.reset()
        self.serow_framework.set_state(self.initial_state)
        return self.initial_state

    def predict_step(self, imu, joint, ft):
        # Process the measurements
        imu, kin, ft = self.serow_framework.process_measurements(imu, joint, ft, None)

        # Predict the base state
        self.serow_framework.base_estimator_predict_step(imu, kin)

        # Get the state
        state = self.serow_framework.get_state(allow_invalid=True)
        return kin, state

    def update_step(self, cf, kin, action, gt, step):
        # Reshape action to (m,1) numpy array
        action = np.array(action, dtype=np.float64).reshape(-1, 1)
        self.serow_framework.set_action(cf, action)
            
        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, kin)

        # Get the post state
        post_state = self.serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        reward, done = self._compute_reward(cf, post_state, gt, step)
        return post_state, reward, done

    def finish_update(self, imu, kin):
        self.serow_framework.base_estimator_finish_update(imu, kin)

    def evaluate(self, observations, agent = None):
        baseline = True
        policy_path = "baseline"
        if agent is not None:
            baseline = False
            policy_path = agent.checkpoint_dir
            # Set to evaluate mode if the agent supports it
            if hasattr(agent, 'eval'):
                agent.eval()

        contact_frames = self.contact_frames
        robot = self.robot

        # After training, evaluate the policy
        print(f"\nEvaluating trained policy for {robot} from {policy_path}...")
        max_steps = len(observations['imu']) - 1
        
        # Get the measurements and the ground truth
        imu_measurements = observations['imu'][:max_steps]
        joint_measurements = observations['joints'][:max_steps]
        force_torque_measurements = observations['ft'][:max_steps]
        base_pose_ground_truth = observations['base_pose_ground_truth'][:max_steps]
        contact_states = observations['contact_states'][:max_steps]

        # Run SEROW
        timestamps = []
        base_positions = []
        base_orientations = []
        cumulative_rewards = {cf: [] for cf in contact_frames}
        immediate_rewards = {cf: [] for cf in contact_frames}
        gt_positions = []
        gt_orientations = []
        gt_timestamps = []

        for step, (imu, joints, ft, gt, cs) in enumerate(zip(imu_measurements, 
                                                             joint_measurements, 
                                                             force_torque_measurements, 
                                                             base_pose_ground_truth,
                                                             contact_states)):
            rewards = {cf: None for cf in contact_frames}

            # Run the predict step
            kin, prior_state = self.predict_step(imu, joints, ft)

            for cf in contact_frames:
                if not baseline and prior_state.get_contact_position(cf) is not None:
                    # Compute the state
                    x = self.compute_state(cf, prior_state, cs)

                    # Compute the action
                    action, _, _ = agent.get_action(x, deterministic=True)
                else:
                    action = np.ones(self.action_dim)

                # Run the update step
                post_state, reward, _ = self.update_step(cf, kin, action, gt, step)
                rewards[cf] = reward

                # Update the prior state
                prior_state = post_state

            # Finish the update
            self.finish_update(imu, kin)        
            
            # Store the data
            timestamps.append(imu.timestamp)
            base_positions.append(post_state.get_base_position())
            base_orientations.append(post_state.get_base_orientation())
            gt_positions.append(gt.position)
            gt_orientations.append(gt.orientation)
            gt_timestamps.append(gt.timestamp)

            # Compute the rewards
            for cf in contact_frames:
                if rewards[cf] is not None:
                    immediate_rewards[cf].append(rewards[cf])
                    if len(cumulative_rewards[cf]) == 0:
                        cumulative_rewards[cf].append(rewards[cf])
                    else:
                        cumulative_rewards[cf].append(cumulative_rewards[cf][-1] + rewards[cf])

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        base_positions = np.array(base_positions)
        base_orientations = np.array(base_orientations)
        gt_positions = np.array(gt_positions)
        gt_orientations = np.array(gt_orientations)
        cumulative_rewards = {cf: np.array(cumulative_rewards[cf]) for cf in contact_frames}

        # Sync and align the data
        timestamps, base_positions, base_orientations, gt_positions, gt_orientations = \
            sync_and_align_data(timestamps, base_positions, base_orientations, gt_timestamps, 
                                gt_positions, gt_orientations, align=True)

        # Plot the trajectories
        plot_trajectories(timestamps, base_positions, base_orientations, gt_positions, 
                          gt_orientations, cumulative_rewards)

        # Print evaluation metrics
        print("\n Policy Evaluation Metrics:")
        for cf in contact_frames:
            print(f"Average Reward for {cf}: {np.mean(immediate_rewards[cf]):.4f}")
            print(f"Max Reward for {cf}: {np.max(immediate_rewards[cf]):.4f} at step " 
                  f"{np.argmax(immediate_rewards[cf])}")
            print(f"Min Reward for {cf}: {np.min(immediate_rewards[cf]):.4f} at step " 
                  f"{np.argmin(immediate_rewards[cf])}")
            print("-------------------------------------------------")
        return timestamps, base_positions, base_orientations, gt_positions, gt_orientations, cumulative_rewards
