import numpy as np
import serow

from env import SerowEnv
from read_mcap import(
    read_base_states, 
    read_contact_states, 
    read_force_torque_measurements, 
    read_joint_measurements, 
    read_imu_measurements, 
    read_base_pose_ground_truth,
    read_joint_states
)


def generate_training_dataset(robot, mcap_path):
    # Load and preprocess the data
    imu_measurements  = read_imu_measurements(mcap_path + "/serow_measurements.mcap")
    joint_measurements = read_joint_measurements(mcap_path + "/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements(mcap_path + "/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth(mcap_path + "/serow_measurements.mcap")
    base_states = read_base_states(mcap_path + "/serow_proprioception.mcap")
    contact_states = read_contact_states(mcap_path + "/serow_proprioception.mcap")
    joint_states = read_joint_states(mcap_path + "/serow_proprioception.mcap")

    # Compute the dt
    dt = []
    dataset_size = len(imu_measurements) - 1    
    for i in range(dataset_size):
        dt.append(imu_measurements[i+1].timestamp - imu_measurements[i].timestamp)
    dt = np.median(np.array(dt))
    print(f"dt: {dt}")

    # Define the dimensions of your state and action spaces
    history_buffer_size = 10
    action_dim = 3  # Based on the action vector used in ContactEKF.setAction()
    state_dim = 3 + 3 * 3 + 3 * 3 * history_buffer_size + 3 * history_buffer_size
    history_buffer_size = 10

    normalizer = None
    serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0], action_dim, 
                         state_dim, history_buffer_size, normalizer)

    dataset = {
            'imu': imu_measurements,
            'joints': joint_measurements,
            'ft': force_torque_measurements,
            'base_states': base_states,
            'contact_states': contact_states,
            'joint_states': joint_states,
            'base_pose_ground_truth': base_pose_ground_truth
        }
        
    timestamps, _, _, gt_positions, gt_orientations, _, kinematics = serow_env.evaluate(dataset, 
                                                                                        agent=None)

    dataset['kinematics'] = kinematics

    # Reform the ground truth data
    base_pose_ground_truth = []
    for i in range(len(timestamps)):
        gt = serow.BasePoseGroundTruth()
        gt.timestamp = timestamps[i]
        gt.position = gt_positions[i]
        gt.orientation = gt_orientations[i]
        base_pose_ground_truth.append(gt)
    dataset['base_pose_ground_truth'] = base_pose_ground_truth

    # Save dataset to a numpy file
    dataset_path = robot + '_training_dataset.npz'
    np.savez(dataset_path,
            imu=dataset['imu'],
            joints=dataset['joints'], 
            ft=dataset['ft'],
            base_states=dataset['base_states'],
            contact_states=dataset['contact_states'],
            joint_states=dataset['joint_states'],
            base_pose_ground_truth=dataset['base_pose_ground_truth'],
            kinematics=dataset['kinematics'],
            dt=dt)

    # print(f"Dataset saved to {dataset_path}")
    # # Load the dataset and verify the data
    # print(f"\nLoading and verifying dataset from {dataset_path}...")

    # # Load the saved dataset
    # loaded_data = np.load(dataset_path, allow_pickle=True)
    # print(loaded_data)
    # print(loaded_data['kinematics'][0])
    # print(loaded_data['base_pose_ground_truth'][0])
    # print(loaded_data['joint_states'][0])
    # print(loaded_data['contact_states'][0])
    # print(loaded_data['base_states'][0])
    # print(loaded_data['ft'][0])
    # print(loaded_data['joints'][0])
    # print(loaded_data['imu'][0])

if __name__ == "__main__":
    generate_training_dataset("go2", "/tmp")
