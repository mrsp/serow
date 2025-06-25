import serow
import numpy as np

from utils import quaternion_to_rotation_matrix, logMap, sync_and_align_data, plot_trajectories

class SerowEnv:
    def __init__(self, robot, joint_state, base_state, contact_state, action_dim, state_dim, dt):
        self.robot = robot
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{robot}_rl.json")
        self.initial_state = self.serow_framework.get_state(allow_invalid=True)
        self.initial_state.set_joint_state(joint_state)
        self.initial_state.set_base_state(base_state)  
        self.initial_state.set_contact_state(contact_state)
        self.serow_framework.set_state(self.initial_state)
        self.contact_frames = self.initial_state.get_contacts_frame()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.previous_position_error_ = 0.0
        self.previous_orientation_error_ = 0.0
        self.dt = dt

    def compute_reward(self, state, gt, step, max_steps):
        reward = None
        done = None

        # Position error
        position_error = np.linalg.norm(state.get_base_position() - gt.position)
        # Orientation error
        orientation_error = np.linalg.norm(logMap(quaternion_to_rotation_matrix(gt.orientation).transpose() 
                                                   @ quaternion_to_rotation_matrix(state.get_base_orientation())))
        position_improvement = max(0, (self.previous_position_error_ - position_error) / self.dt)
        self.previous_position_error_ = position_error

        orientation_improvement = max(0, (self.previous_orientation_error_ - orientation_error) / self.dt)
        self.previous_orientation_error_ = orientation_error

        if (np.linalg.norm(position_error) > 0.15 or np.linalg.norm(orientation_error) > 0.1):
            done = 1.0  
            reward = -100.0
            position_reward = 0.0
            orientation_reward = 0.0
        else:
            done = 0.0
            position_reward = position_improvement
            orientation_reward = orientation_improvement
            reward = position_reward + orientation_reward + (step + 1) / max_steps
        
        # Reward diagnostics
        # print(f"[Reward Debug] step={step}, done = {done}, pos_reward={position_reward}, ori_reward={orientation_reward}, total_reward={reward}")

        return reward, done    

    def compute_state(self, cf, state, kin):
        R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
        local_pos = R_base @ (state.get_contact_position(cf) - state.get_base_position())   
        local_kin_pos = kin.contacts_position[cf]
        R = kin.contacts_position_noise[cf] + kin.position_cov
        e = np.absolute(local_pos - local_kin_pos)
        e = np.linalg.inv(R) @ e
        return np.concatenate((e, np.array([kin.contacts_probability[cf]])), axis=0)

    def reset(self):
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{self.robot}_rl.json")
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

    def update_step(self, cf, kin, action, gt, step, max_steps):
        # Reshape action to (m,1) numpy array
        action = np.array(action, dtype=np.float64).reshape(-1, 1)
        self.serow_framework.set_action(cf, action)
            
        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, kin)

        # Get the post state
        post_state = self.serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        reward, done = self.compute_reward(post_state, gt, step, max_steps)
        return post_state, reward, done

    def finish_update(self, imu, kin):
        self.serow_framework.base_estimator_finish_update(imu, kin)

    def evaluate(self, observations, agent = None):
        self.reset()

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

        # Run SEROW
        timestamps = []
        base_positions = []
        base_orientations = []
        cumulative_rewards = {cf: [] for cf in contact_frames}
        immediate_rewards = {cf: [] for cf in contact_frames}
        gt_positions = []
        gt_orientations = []
        gt_timestamps = []
        kinematics = []

        for step, (imu, joints, ft, gt) in enumerate(zip(imu_measurements, 
                                                         joint_measurements, 
                                                         force_torque_measurements, 
                                                         base_pose_ground_truth)):
            rewards = {cf: None for cf in contact_frames}

            # Run the predict step
            kin, prior_state = self.predict_step(imu, joints, ft)

            for cf in contact_frames:
                if not baseline and prior_state.get_contact_position(cf) is not None:
                    # Compute the state
                    x = self.compute_state(cf, prior_state, kin)

                    # Compute the action    
                    action = agent.get_action(x, deterministic=True)[0]
                else:
                    action = np.zeros(self.action_dim)

                # Run the update step
                post_state, reward, _ = self.update_step(cf, kin, action, gt, step, max_steps)
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
            kinematics.append(kin)

            # Compute the rewards
            for cf in contact_frames:
                if rewards[cf] is not None:
                    # Ensure reward is a scalar value
                    reward_value = float(rewards[cf])
                    immediate_rewards[cf].append(reward_value)
                    if len(cumulative_rewards[cf]) == 0:
                        cumulative_rewards[cf].append(reward_value)
                    else:
                        cumulative_rewards[cf].append(cumulative_rewards[cf][-1] + reward_value)

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        base_positions = np.array(base_positions)
        base_orientations = np.array(base_orientations)
        gt_positions = np.array(gt_positions)
        gt_orientations = np.array(gt_orientations)
        cumulative_rewards = {cf: np.array(cumulative_rewards[cf]) for cf in contact_frames}
        immediate_rewards = {cf: np.array(immediate_rewards[cf]) for cf in contact_frames}

        # Sync and align the data
        timestamps, base_positions, base_orientations, gt_positions, gt_orientations = \
            sync_and_align_data(timestamps, base_positions, base_orientations, gt_timestamps, 
                                gt_positions, gt_orientations, align=True)

        # Plot the trajectories
        plot_trajectories(timestamps, base_positions, base_orientations, gt_positions, 
                          gt_orientations)
        
        # Print evaluation metrics
        print("\n Policy Evaluation Metrics:")
        for cf in contact_frames:
            print(f"Average Reward for {cf}: {np.mean(immediate_rewards[cf])}")
            print(f"Max Reward for {cf}: {np.max(immediate_rewards[cf])} at step " 
                  f"{np.argmax(immediate_rewards[cf])}")
            print(f"Min Reward for {cf}: {np.min(immediate_rewards[cf])} at step " 
                  f"{np.argmin(immediate_rewards[cf])}")
            print("-------------------------------------------------")
        return timestamps, base_positions, base_orientations, gt_positions, gt_orientations, cumulative_rewards, kinematics
