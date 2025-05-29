# Visualizing data for SEROW

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import json

display_plots = True
imu_calibration_cycles = 0

config_file = 'test_config.json'  # Path to your JSON config file
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file {config_file} not found.")

with open(config_file, 'r') as f:
    config = json.load(f)
    
serow_path = os.environ.get("SEROW_PATH")
if not serow_path:
    raise EnvironmentError("SEROW_PATH environment variable not set.")

# Resolve paths from the JSON configuration
base_path = config['Paths']['base_path']
experiment_type =  config["Experiment"]["type"]
measurement_file = serow_path +  config["Paths"]["data_file"].replace("{base_path}", base_path).replace("{type}", experiment_type)
prediction_file = serow_path + config['Paths']['prediction_file'].replace("{base_path}", base_path).replace("{type}", experiment_type)


# Load the data from the HDF5 file
def load_gt_data(h5_file):
    with h5py.File(h5_file, "r") as f:
        positions = np.array(f["/base_ground_truth/position"])
        orientations = np.array(f["/base_ground_truth/orientation"])
        FL_forces = np.array(f["/feet_force/FL"])
        FR_forces = np.array(f["/feet_force/FR"])
        RL_forces = np.array(f["/feet_force/RL"])
        RR_forces = np.array(f["/feet_force/RR"])

        imu = np.array(f["/imu/linear_acceleration"])
        
    return positions, orientations, imu, FL_forces,FR_forces,RL_forces,RR_forces

def print_meta(h5_file):
    with h5py.File(h5_file, "r") as f:
        for name in f.keys():
            print(name)
            
def load_serow_preds(h5_file):
    with h5py.File(h5_file, "r") as f:
        timestamps = np.array(f["timestamp/t"])
        pos_x = np.array(f["base_pose/position/x"])
        pos_y = np.array(f["base_pose/position/y"])
        pos_z = np.array(f["base_pose/position/z"])
        
        rot_x = np.array(f["base_pose/rotation/x"])
        rot_y = np.array(f["base_pose/rotation/y"])
        rot_z = np.array(f["base_pose/rotation/z"])
        rot_w = np.array(f["base_pose/rotation/w"])
        
        b_ax = np.array(f["imu_bias/accel/x"])
        b_ay = np.array(f["imu_bias/accel/y"])
        b_az = np.array(f["imu_bias/accel/z"])
        
        b_wx = np.array(f["imu_bias/angVel/x"])
        b_wy = np.array(f["imu_bias/angVel/y"])
        b_wz = np.array(f["imu_bias/angVel/z"])
        
    return timestamps, pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w, b_ax,b_ay,b_az,b_wx,b_wy,b_wz


def compute_ATE_pos(gt_pos, est_x, est_y, est_z):
    est_pos = np.column_stack((est_x, est_y, est_z))
    error = np.linalg.norm(gt_pos-est_pos, axis = 1)
    ate = np.sqrt(np.mean(error**2))
    return ate

def compute_ATE_rot(gt_rot,est_rot_w,est_rot_x,est_rot_y,est_rot_z):
    est_rot = np.column_stack((est_rot_w,est_rot_x, est_rot_y, est_rot_z))
    rotation_errors = np.zeros((gt_rot.shape[0]))
    for i in range(gt_rot.shape[0]):    
        q_gt = gt_rot[i]
        q_est = est_rot[i]
        
        
        q_gt_conj = np.array([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])  # Conjugate of q_gt
        q_rel = quaternion_multiply(q_gt_conj, q_est)
        rotation_errors[i] = 2 * np.arccos(np.clip(q_rel[0],-1.0,1.0))
        
    ate_rot = np.sqrt(np.mean(rotation_errors**2))
    return ate_rot    
            
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 and q2.
    
    Parameters:
    - q1: numpy array of shape (4,), quaternion (w, x, y, z).
    - q2: numpy array of shape (4,), quaternion (w, x, y, z).
    
    Returns:
    - q: numpy array of shape (4,), resulting quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])   

### Removes the bias from the initial pose (world frame is 0 0 0  0 0 0 1 at time t = 0 )
def remove_gt_bias(positions,orientations):
    # Get the initial position and orientation
    initial_position = positions[0]  # Initial position at time t=0
    q0 = orientations[0]  # Initial orientation at time t=0

    # Modify the position and orientation to be relative to the world frame (t=0 as origin)
    # Subtract initial position from all positions to set the initial position to (0, 0, 0)
    positions = positions - initial_position
    # You can modify the orientation similarly by applying the inverse of the initial quaternion.
    # Assuming orientations are in quaternion format (x, y, z, w), we need to inverse the initial orientation
    initial_orientation_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]]) 
    orientations = np.array([quaternion_multiply(initial_orientation_inv, q) for q in orientations])
    return positions, orientations
    
if __name__ == "__main__":
    gt_pos, gt_rot, imu,FL_forces,FR_forces,RL_forces,RR_forces = load_gt_data(measurement_file)
    timestamps,est_pos_x, est_pos_y, est_pos_z, est_rot_x, est_rot_y, est_rot_z, est_rot_w, b_ax,b_ay,b_az,b_wx,b_wy,b_wz =  load_serow_preds(prediction_file)
    gt_rot = np.column_stack((gt_rot[:, 3], gt_rot[:, 0], gt_rot[:, 1], gt_rot[:, 2]))

    mass = (FL_forces[:1000,2] + FR_forces[:1000,2] + RL_forces[:1000,2] + RR_forces[:1000,2])/9.81
    print("Mass: ", np.mean(mass))
    gt_pos = gt_pos[imu_calibration_cycles:(len(timestamps)+ imu_calibration_cycles)]
    gt_rot = gt_rot[imu_calibration_cycles:(len(timestamps)+ imu_calibration_cycles)]
    
    FL_forces = FL_forces[imu_calibration_cycles:(len(timestamps)+imu_calibration_cycles)]
    FR_forces = FR_forces[imu_calibration_cycles:(len(timestamps)+imu_calibration_cycles)]
    RL_forces = RL_forces[imu_calibration_cycles:(len(timestamps)+imu_calibration_cycles)]
    RR_forces = RR_forces[imu_calibration_cycles:(len(timestamps)+imu_calibration_cycles)]
    print("Base position ATE: ", compute_ATE_pos(gt_pos,est_pos_x,est_pos_y,est_pos_z))
    print("Base rotation ATE: ", compute_ATE_rot(gt_rot,est_rot_w, est_rot_x,est_rot_y,est_rot_z))
    gt_pos, gt_rot = remove_gt_bias(gt_pos,gt_rot)
    # com_vel = compute_com_vel(com_pos,timestamps)
    
   
    # Plotting Ground Truth and Estimated Position (x, y, z)
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig1.suptitle("Base position")

    axs1[0].plot(timestamps, gt_pos[:,0], label='Ground Truth', color='blue')
    axs1[0].plot(timestamps, est_pos_x, label='Estimated', color='orange', linestyle='--')
    axs1[0].set_ylabel('base_pos_x')
    axs1[0].legend()

    axs1[1].plot(timestamps, gt_pos[:,1], label='Ground Truth', color='blue')
    axs1[1].plot(timestamps, est_pos_y, label='Estimated', color='orange', linestyle='--')
    axs1[1].set_ylabel('base_pos_y')
    axs1[1].legend()

    axs1[2].plot(timestamps, gt_pos[:,2], label='Ground Truth', color='blue')
    axs1[2].plot(timestamps, est_pos_z, label='Estimated', color='orange', linestyle='--')
    axs1[2].set_ylabel('base_pos_z')
    axs1[2].set_xlabel('Timestamp')
    axs1[2].legend()


    # Plotting Ground Truth and Estimated Orientation
    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig2.suptitle("Base Orientation")

    axs2[0].plot(timestamps, gt_rot[:,1], label='Ground Truth', color='blue')
    axs2[0].plot(timestamps, est_rot_x, label='Estimated', color='orange', linestyle='--')
    axs2[0].set_ylabel('Orientation W')
    axs2[0].legend()

    axs2[1].plot(timestamps, gt_rot[:,2], label='Ground Truth', color='blue')
    axs2[1].plot(timestamps, est_rot_y, label='Estimated', color='orange', linestyle='--')
    axs2[1].set_ylabel('Orientation X')
    axs2[1].legend()

    axs2[2].plot(timestamps, gt_rot[:,3], label='Ground Truth', color='blue')
    axs2[2].plot(timestamps, est_rot_z, label='Estimated', color='orange', linestyle='--')
    axs2[2].set_ylabel('Orientation Y')
    axs2[2].legend()

    axs2[3].plot(timestamps, gt_rot[:,0], label='Ground Truth', color='blue')
    axs2[3].plot(timestamps, est_rot_w, label='Estimated', color='orange', linestyle='--')
    axs2[3].set_ylabel('Orientation Z')
    axs2[3].set_xlabel('Timestamp')
    axs2[3].legend()

    # Plotting Ground Truth and Estimated Position (x, y, z)
    fig4, axs4 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig4.suptitle("Feet forces (z-axis only)")

    axs4[0].plot(timestamps, FL_forces[:,2], label='Ground Truth', color='blue')
    axs4[0].set_ylabel('FORCE FL')

    axs4[1].plot(timestamps, FR_forces[:,2], label='Ground Truth', color='blue')
    axs4[1].set_ylabel('FORCE FR')

    axs4[2].plot(timestamps, RL_forces[:,2], label='Ground Truth', color='blue')
    axs4[2].set_ylabel('FORCE RL')

    axs4[3].plot(timestamps, RR_forces[:,2], label='Ground Truth', color='blue')
    axs4[3].set_ylabel('FORCE RR')
    axs4[3].set_xlabel('Timestamp')
    
    # Plotting Ground Truth and Estimated Position (x, y, z)
    fig5, axs5 = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    fig5.suptitle("IMU estimated biases")

    axs5[0].plot(timestamps, b_ax, label='Ground Truth', color='blue')
    axs5[0].set_ylabel('bias ax')

    axs5[1].plot(timestamps,b_ay, color='blue')
    axs5[1].set_ylabel('bias ay')

    axs5[2].plot(timestamps, b_az, color='blue')
    axs5[2].set_ylabel('bias az')
 
    axs5[3].plot(timestamps,b_wx, color='blue')
    axs5[3].set_ylabel('bias wx')
    
    axs5[4].plot(timestamps,b_wy, color='blue')
    axs5[4].set_ylabel('bias wy')
    
    
    axs5[5].plot(timestamps, b_wz, color='blue')
    axs5[5].set_ylabel('bias wz')
    
    axs5[3].set_xlabel('Timestamp')

    plt.tight_layout()

    if (display_plots):
        plt.show()
