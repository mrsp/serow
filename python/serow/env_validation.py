import numpy as np
import serow
import matplotlib.pyplot as plt
import unittest
import zipfile
import os


def run_serow(dataset, robot):
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}_pytest.json")

    base_positions = []
    base_orientations = []
    for imu, joint, ft in zip(dataset["imu"], dataset["joints"], dataset["ft"]):
        status = serow_framework.filter(imu, joint, ft, None)
        if status:
            state = serow_framework.get_state(allow_invalid=True)
            base_positions.append(state.get_base_position())
            base_orientations.append(state.get_base_orientation())

    return np.array(base_positions), np.array(base_orientations)

def run_serow_per_step(dataset, robot):
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}_pytest.json")

    base_positions = []
    base_orientations = []
    for imu, joint, ft in zip(dataset["imu"], dataset["joints"], dataset["ft"]):
        imu, kin, ft = serow_framework.process_measurements(imu, joint, ft, None)
        serow_framework.base_estimator_predict_step(imu, kin)        
        for cf in kin.contacts_status.keys():
            serow_framework.base_estimator_update_with_contact_position(cf, kin)
        serow_framework.base_estimator_finish_update(imu, kin)
        state = serow_framework.get_state(allow_invalid=True)
        base_positions.append(state.get_base_position())
        base_orientations.append(state.get_base_orientation())
    return np.array(base_positions), np.array(base_orientations)


def run_serow_playback(dataset):
    base_positions = []
    base_orientations = []
    for bs in dataset["base_states"]:
        base_positions.append(bs.base_position)
        base_orientations.append(bs.base_orientation)

    return np.array(base_positions), np.array(base_orientations)


class TestSerow(unittest.TestCase):
    def setUp(self):
        self.robot = "go2"
        # Unzip the dataset if the unzipped file does not exist
        if not os.path.exists(f"{self.robot}_test.npz"):
            with zipfile.ZipFile(f"{self.robot}_test.zip", 'r') as zip_ref:
                zip_ref.extractall()
        self.dataset = np.load(f"{self.robot}_test.npz", allow_pickle=True)
        print(f"Length of dataset base states: {len(self.dataset['base_states'])}")
        print(f"Length of dataset joint states: {len(self.dataset['joints'])}")
        print(f"Length of dataset imu: {len(self.dataset['imu'])}")
        print(f"Length of dataset ft: {len(self.dataset['ft'])}")
        print(f"Length of dataset base pose ground truth: {len(self.dataset['base_pose_ground_truth'])}")
    def test_serow(self):
        base_positions, base_orientations = run_serow(self.dataset, self.robot)
        actual_base_positions, actual_base_orientations = run_serow_playback(
            self.dataset
        )
        base_positions_per_step, base_orientations_per_step = run_serow_per_step(
            self.dataset, self.robot
        )
        print(f"Base positions: {len(base_positions)}")
        print(f"Actual base positions: {len(actual_base_positions)}")
        print(f"Base orientations: {len(base_orientations)}")
        print(f"Actual base orientations: {len(actual_base_orientations)}")
        print(f"Base positions per step: {len(base_positions_per_step)}")
        print(f"Base orientations per step: {len(base_orientations_per_step)}")

        assert len(base_positions) == len(actual_base_positions)
        assert len(base_orientations) == len(actual_base_orientations)
        assert len(base_positions_per_step) == len(actual_base_positions)
        assert len(base_orientations_per_step) == len(actual_base_orientations)

        position_error = base_positions - actual_base_positions
        orientation_error = base_orientations - actual_base_orientations
        position_error_per_step = base_positions_per_step - actual_base_positions
        orientation_error_per_step = base_orientations_per_step - actual_base_orientations
        print(f"Actual Position error: {position_error.sum()}, {position_error.max()}")
        print(
            f"Actual Orientation error: {orientation_error.sum()}, {orientation_error.max()}"
        )
        print(f"Actual Position error per step: {position_error_per_step.sum()}, {position_error_per_step.max()}")
        print(f"Actual Orientation error per step: {orientation_error_per_step.sum()}, {orientation_error_per_step.max()}")

        # Plot the base position and orientation
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(actual_base_positions[:, 0], label="x Actual")
        axs[0].plot(base_positions[:, 0], label="x SEROW")
        axs[0].plot(base_positions_per_step[:, 0], label="x SEROW Per Step")

        axs[0].plot(actual_base_positions[:, 1], label="y Actual")
        axs[0].plot(base_positions[:, 1], label="y SEROW")
        axs[0].plot(base_positions_per_step[:, 1], label="y SEROW Per Step")

        axs[0].plot(actual_base_positions[:, 2], label="z Actual")
        axs[0].plot(base_positions[:, 2], label="z SEROW")
        axs[0].plot(base_positions_per_step[:, 2], label="z SEROW Per Step")

        axs[1].plot(actual_base_orientations[:, 0], label="qw Actual")
        axs[1].plot(base_orientations[:, 0], label="qw SEROW")
        axs[1].plot(base_orientations_per_step[:, 0], label="qw SEROW Per Step")

        axs[1].plot(actual_base_orientations[:, 1], label="qx Actual")
        axs[1].plot(base_orientations[:, 1], label="qx SEROW")
        axs[1].plot(base_orientations_per_step[:, 1], label="qx SEROW Per Step")

        axs[1].plot(actual_base_orientations[:, 2], label="qy Actual")
        axs[1].plot(base_orientations[:, 2], label="qy SEROW")
        axs[1].plot(base_orientations_per_step[:, 2], label="qy SEROW Per Step")

        axs[1].plot(actual_base_orientations[:, 3], label="qz Actual")
        axs[1].plot(base_orientations[:, 3], label="qz SEROW")
        axs[1].plot(base_orientations_per_step[:, 3], label="qz SEROW Per Step")

        axs[0].set_title("Base Position")
        axs[1].set_title("Base Orientation")
        axs[0].legend()
        axs[1].legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
