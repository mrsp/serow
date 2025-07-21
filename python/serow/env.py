import serow
import numpy as np
import gymnasium as gym
import copy

from utils import (
    quaternion_to_rotation_matrix,
    logMap,
    sync_and_align_data,
    plot_trajectories,
)


class SerowEnv(gym.Env):
    def __init__(
        self,
        robot,
        contact_frame,
        joint_state,
        base_state,
        contact_state,
        action_dim,
        state_dim,
    ):
        super(SerowEnv, self).__init__()

        # Environment parameters
        self.robot = robot
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{robot}_rl.json")
        self.initial_state = self.serow_framework.get_state(allow_invalid=True)
        self.initial_state.set_joint_state(joint_state)
        self.initial_state.set_base_state(base_state)
        self.initial_state.set_contact_state(contact_state)
        self.serow_framework.set_state(self.initial_state)
        self.cf = contact_frame  # contact frame to control
        self.contact_frames = [
            cf for cf in contact_state.contacts_status.keys() if cf != self.cf
        ]
        self.action_dim = action_dim
        self.state_dim = state_dim

    def _compute_reward(self, cf, state, gt, step, max_steps):
        reward = 0.0
        done = False
        truncated = False
        # Position error
        position_error = np.linalg.norm(state.get_base_position() - gt.position)

        # Orientation error
        orientation_error = np.linalg.norm(
            logMap(
                quaternion_to_rotation_matrix(gt.orientation).transpose()
                @ quaternion_to_rotation_matrix(state.get_base_orientation())
            )
        )

        # Compute innovation and S
        success, innovation, covariance = (
            self.serow_framework.get_contact_position_innovation(cf)
        )

        max_position_error = 3.0
        max_orientation_error = 1.0
        if (
            position_error > max_position_error
            or orientation_error > max_orientation_error
        ):
            done = True
        else:
            if success:
                nis = innovation @ np.linalg.inv(covariance) @ innovation.T
                nis_reward = -nis * 50
                position_reward = -position_error / max_position_error
                orientation_reward = -orientation_error / max_orientation_error
                step_reward = 1.0 * (step + 1) / max_steps
                reward = nis_reward + 2.0 * position_reward + 3.0 * orientation_reward
                reward /= 3.0  # number of terms
                reward += step_reward
                if self.step_count == self.max_steps - 1:
                    truncated = True
                    done = False

        return reward, done, truncated

    def _get_observation(self, cf, state, kin):
        R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
        local_pos = R_base @ (
            state.get_contact_position(cf) - state.get_base_position()
        )
        local_kin_pos = kin.contacts_position[cf]
        innovation = local_kin_pos - local_pos
        R = (kin.contacts_position_noise[cf] + kin.position_cov).flatten()
        P_pos_trace = np.trace(state.get_base_position_cov())
        P_ori_trace = np.trace(state.get_base_orientation_cov())

        return np.concatenate(
            [
                [P_pos_trace],
                [P_ori_trace],
                innovation,
                R,
            ],
            axis=0,
        )

    def reset(self):
        self.serow_framework.reset()
        self.serow_framework.initialize(f"{self.robot}_rl.json")
        self.serow_framework.set_state(self.initial_state)
        self.step_count = 0
        return self.initial_state

    def pre_step(self):
        result = self.predict_step(self.imu, self.joint, self.ft)
        if result is not None:
            prior_state, kin = result

    def step(self, action):
        imu = copy.copy(self.imu_data[self.step_count])
        joint = copy.copy(self.joint_data[self.step_count])
        ft = copy.copy(self.ft_data[self.step_count])
        gt = copy.copy(self.gt_data[self.step_count])
        reward = 0.0
        done = False
        truncated = False

        result = self.predict_step(imu, joint, ft)
        if result is not None:
            prior_state, kin = result
        else:
            return

        obs = None
        if kin.contacts_status[self.cf]:
            post_state, reward, done, truncated = self.update_step(
                self.cf,
                kin,
                action,
                gt,
                self.step_count,
                self.max_steps,
            )

            obs = self._get_observation(self.cf, post_state, kin)
        info = {
            "step_count": self.step_count,
            "reward": reward,
        }

        for cf in self.contact_frames:
            if kin.contacts_status[cf]:
                post_state, _, _, _ = self.update_step(
                    cf, kin, action, gt, self.step_count, self.max_steps
                )

        self.finish_update(imu, kin)
        self.step_count += 1

        return obs, reward, done, truncated, info

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
        return state, kin

    def update_step(self, cf, kin, action, gt, step, max_steps):
        # Set the action
        self.serow_framework.set_action(cf, action)

        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, kin)

        # Get the post state
        post_state = self.serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        reward, done, truncated = self._compute_reward(
            cf, post_state, gt, step, max_steps
        )

        return post_state, reward, done, truncated

    def finish_update(self, imu, kin):
        self.serow_framework.base_estimator_finish_update(imu, kin)

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.step_count}")

    def evaluate(self, observations):
        self.reset()

        # After training, evaluate the policy
        max_steps = len(observations["imu"]) - 1
        self.max_steps = max_steps

        # Get the measurements and the ground truth
        self.imu_data = observations["imu"][:max_steps]
        self.joint_data = observations["joints"][:max_steps]
        self.ft_data = observations["ft"][:max_steps]
        self.gt_data = observations["base_pose_ground_truth"][:max_steps]

        # Run SEROW
        timestamps = []
        base_positions = []
        base_orientations = []
        cumulative_rewards = []
        immediate_rewards = []
        gt_positions = []
        gt_orientations = []
        gt_timestamps = []

        for _ in range(max_steps):
            _, reward, _, _, _ = self.step(
                np.zeros((self.action_dim, 1), dtype=np.float64)
            )
            post_state = self.serow_framework.get_state(allow_invalid=True)

            # Store the data
            immediate_rewards.append(reward)
            if len(cumulative_rewards) == 0:
                cumulative_rewards.append(reward)
            else:
                cumulative_rewards.append(cumulative_rewards[-1] + reward)
            timestamps.append(self.imu_data[self.step_count - 1].timestamp)
            gt_positions.append(self.gt_data[self.step_count - 1].position)
            gt_orientations.append(self.gt_data[self.step_count - 1].orientation)
            gt_timestamps.append(self.gt_data[self.step_count - 1].timestamp)
            base_positions.append(post_state.get_base_position())
            base_orientations.append(post_state.get_base_orientation())

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        base_positions = np.array(base_positions)
        base_orientations = np.array(base_orientations)
        gt_positions = np.array(gt_positions)
        gt_orientations = np.array(gt_orientations)

        # Sync and align the data
        (
            timestamps,
            base_positions,
            base_orientations,
            gt_positions,
            gt_orientations,
        ) = sync_and_align_data(
            timestamps,
            base_positions,
            base_orientations,
            gt_timestamps,
            gt_positions,
            gt_orientations,
            align=True,
        )

        # Plot the trajectories
        plot_trajectories(
            timestamps,
            base_positions,
            base_orientations,
            gt_positions,
            gt_orientations,
        )
        return (
            timestamps,
            base_positions,
            base_orientations,
            gt_positions,
            gt_orientations,
            cumulative_rewards,
        )
