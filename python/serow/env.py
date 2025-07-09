from unittest import result
import serow
import numpy as np

from utils import (
    quaternion_to_rotation_matrix,
    logMap,
    sync_and_align_data,
    plot_trajectories,
)


class SerowEnv:
    def __init__(
        self,
        robot,
        joint_state,
        base_state,
        contact_state,
        action_dim,
        state_dim,
        history_buffer_size,
        state_normalizer=None,
    ):
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
        self.R_history_buffer = []
        self.R_initial = np.eye(3, dtype=np.float64)
        self.R_history_buffer_size = history_buffer_size
        self.innovation_history_buffer = []
        self.innovation_history_buffer_size = history_buffer_size
        self.innovation_initial = np.zeros(3, dtype=np.float64)
        self.state_normalizer = state_normalizer

    def get_R_history(self):
        max_history_size = self.R_history_buffer_size

        if len(self.R_history_buffer) == 0:
            # Fill entire history with initial R
            R_history_filled = [self.R_initial] * max_history_size
        elif len(self.R_history_buffer) < max_history_size:
            # Fill missing entries with initial R
            padding_needed = max_history_size - len(self.R_history_buffer)
            R_history_filled = [self.R_initial] * padding_needed + self.R_history_buffer
        else:
            R_history_filled = self.R_history_buffer[-max_history_size:]

        return np.stack(R_history_filled).flatten()

    def get_innovation_history(self):
        max_history_size = self.innovation_history_buffer_size

        if len(self.innovation_history_buffer) == 0:
            innovation_history_filled = [self.innovation_initial] * max_history_size
        elif len(self.innovation_history_buffer) < max_history_size:
            padding_needed = max_history_size - len(self.innovation_history_buffer)
            innovation_history_filled = [
                self.innovation_initial
            ] * padding_needed + self.innovation_history_buffer
        else:
            innovation_history_filled = self.innovation_history_buffer[
                -max_history_size:
            ]
        return np.stack(innovation_history_filled).flatten()

    def compute_reward(self, cf, state, gt, step, max_steps):
        reward = 0.0
        done = 0.0

        # Position error
        position_error = np.linalg.norm(state.get_base_position() - gt.position)

        # Orientation error
        orientation_error = np.linalg.norm(
            logMap(
                quaternion_to_rotation_matrix(gt.orientation).transpose()
                @ quaternion_to_rotation_matrix(state.get_base_orientation())
            )
        )

        success, _, _, innovation, covariance = (
            self.serow_framework.get_contact_position_innovation(cf)
        )

        if (
            np.linalg.norm(position_error) > 0.35
            or np.linalg.norm(orientation_error) > 0.1
        ):
            done = 1.0
            reward = -50.0
            position_reward = 0.0
            orientation_reward = 0.0
        else:
            if success:
                done = 0.0
                nis = innovation @ np.linalg.inv(covariance) @ innovation.T
                nis_reward = np.exp(-100.0 * nis)
                position_reward = np.exp(-10.0 * position_error)
                orientation_reward = np.exp(-50.0 * orientation_error)
                step_reward = 1.0 * (step + 1) / max_steps

                reward = nis_reward + position_reward + orientation_reward
                reward /= 3.0
                reward += step_reward

        return reward, done, innovation

    def compute_state(self, cf, state, kin):
        R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
        local_pos = R_base @ (
            state.get_contact_position(cf) - state.get_base_position()
        )
        local_kin_pos = kin.contacts_position[cf]
        R = kin.contacts_position_noise[cf] + kin.position_cov
        innovation = local_kin_pos - local_pos

        if self.state_normalizer is not None:
            innovation = self.state_normalizer.normalize_innovation(innovation)
            R = self.state_normalizer.normalize_R(R)

        R_history = self.get_R_history()
        innovation_history = self.get_innovation_history()
        P_pos_trace = np.trace(state.get_base_position_cov())
        P_ori_trace = np.trace(state.get_base_orientation_cov())
        return np.concatenate(
            [
                [P_pos_trace],
                [P_ori_trace],
                innovation,
                R.flatten(),
                innovation_history,
                R_history,
            ],
            axis=0,
        )

    def reset(self):
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{self.robot}_rl.json")
        self.serow_framework.set_state(self.initial_state)

        self.innovation_initial = np.zeros(3, dtype=np.float64)
        if len(self.innovation_history_buffer) > 0:
            for i in range(0, len(self.innovation_history_buffer)):
                self.innovation_initial += self.innovation_history_buffer[i]
            self.innovation_initial /= len(self.innovation_history_buffer)

        self.R_initial = np.eye(3, dtype=np.float64)
        if len(self.R_history_buffer) > 0:
            self.R_initial = np.zeros((3, 3), dtype=np.float64)
            for i in range(0, len(self.R_history_buffer)):
                self.R_initial += self.R_history_buffer[i]
            self.R_initial /= len(self.R_history_buffer)
        self.R_history_buffer = []
        self.innovation_history_buffer = []
        return self.initial_state

    def predict_step(self, imu, joint, ft):
        # Process the measurements
        result = self.serow_framework.process_measurements(imu, joint, ft, None)

        if result is None:
            return None, None
        imu, kin, ft = result

        # Predict the base state
        self.serow_framework.base_estimator_predict_step(imu, kin)

        # Get the state
        state = self.serow_framework.get_state(allow_invalid=True)
        return kin, state

    def update_step(self, cf, kin, action, gt, step, max_steps):
        # Set the action
        self.serow_framework.set_action(cf, action)

        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, kin)

        # Get the post state
        post_state = self.serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        reward, done, innovation = self.compute_reward(
            cf, post_state, gt, step, max_steps
        )

        if self.state_normalizer is not None:
            innovation = self.state_normalizer.normalize_innovation(innovation)

        while (
            len(self.innovation_history_buffer) >= self.innovation_history_buffer_size
        ):
            self.innovation_history_buffer.pop(0)
        self.innovation_history_buffer.append(innovation)

        # Update the R history buffer
        R = kin.contacts_position_noise[cf] + kin.position_cov
        if np.any(action != np.zeros(self.action_dim)):
            L = np.zeros((3, 3), dtype=np.float64)
            L[0, 0] = action[0]
            L[1, 1] = action[1]
            L[2, 2] = action[2]
            L[1, 0] = action[3]
            L[2, 0] = action[4]
            L[2, 1] = action[5]
            # Reconstruct the matrix
            R = L @ L.T

        if self.state_normalizer is not None:
            R = self.state_normalizer.normalize_R_with_action(R)

        while len(self.R_history_buffer) >= self.R_history_buffer_size:
            self.R_history_buffer.pop(0)
        self.R_history_buffer.append(R)

        return post_state, reward, done

    def finish_update(self, imu, kin):
        self.serow_framework.base_estimator_finish_update(imu, kin)

    def evaluate(self, observations, agent=None):
        self.reset()

        baseline = True
        policy_path = "baseline"
        if agent is not None:
            baseline = False
            policy_path = agent.checkpoint_dir
            # Set to evaluate mode if the agent supports it
            if hasattr(agent, "eval"):
                agent.eval()

        contact_frames = self.contact_frames
        robot = self.robot

        # After training, evaluate the policy
        print(f"\nEvaluating trained policy for {robot} from {policy_path}...")
        max_steps = len(observations["imu"]) - 1

        # Get the measurements and the ground truth
        imu_measurements = observations["imu"][:max_steps]
        joint_measurements = observations["joints"][:max_steps]
        force_torque_measurements = observations["ft"][:max_steps]
        base_pose_ground_truth = observations["base_pose_ground_truth"][:max_steps]

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

        for step, (imu, joints, ft, gt) in enumerate(
            zip(
                imu_measurements,
                joint_measurements,
                force_torque_measurements,
                base_pose_ground_truth,
            )
        ):
            rewards = {cf: None for cf in contact_frames}

            # Run the predict step
            kin, prior_state = self.predict_step(imu, joints, ft)

            if kin is None or prior_state is None:
                continue

            post_state = prior_state
            for cf in contact_frames:
                if not baseline and post_state.get_contact_position(cf) is not None:
                    # Compute the state
                    x = self.compute_state(cf, post_state, kin)

                    # Compute the action
                    if agent.name == "DDPG":
                        action = agent.get_action(x, deterministic=True)
                    else:
                        action, _, _ = agent.get_action(x, deterministic=True)
                else:
                    action = np.zeros((self.action_dim, 1), dtype=np.float64)

                # Run the update step
                post_state, reward, _ = self.update_step(
                    cf, kin, action, gt, step, max_steps
                )
                rewards[cf] = reward

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
                        cumulative_rewards[cf].append(
                            cumulative_rewards[cf][-1] + reward_value
                        )

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        base_positions = np.array(base_positions)
        base_orientations = np.array(base_orientations)
        gt_positions = np.array(gt_positions)
        gt_orientations = np.array(gt_orientations)
        cumulative_rewards = {
            cf: np.array(cumulative_rewards[cf]) for cf in contact_frames
        }
        immediate_rewards = {
            cf: np.array(immediate_rewards[cf]) for cf in contact_frames
        }

        # Sync and align the data
        timestamps, base_positions, base_orientations, gt_positions, gt_orientations = (
            sync_and_align_data(
                timestamps,
                base_positions,
                base_orientations,
                gt_timestamps,
                gt_positions,
                gt_orientations,
                align=True,
            )
        )

        # Plot the trajectories
        plot_trajectories(
            timestamps, base_positions, base_orientations, gt_positions, gt_orientations
        )

        # Print evaluation metrics
        print("\n Policy Evaluation Metrics:")
        for cf in contact_frames:
            print(f"Average Reward for {cf}: {np.mean(immediate_rewards[cf])}")
            print(
                f"Max Reward for {cf}: {np.max(immediate_rewards[cf])} at step "
                f"{np.argmax(immediate_rewards[cf])}"
            )
            print(
                f"Min Reward for {cf}: {np.min(immediate_rewards[cf])} at step "
                f"{np.argmin(immediate_rewards[cf])}"
            )
            print("-------------------------------------------------")
        return (
            timestamps,
            base_positions,
            base_orientations,
            gt_positions,
            gt_orientations,
            cumulative_rewards,
            kinematics,
        )
