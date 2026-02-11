#!/usr/bin/env python3
import numpy as np
import serow
import json
import os
import zipfile
from bayes_opt import BayesianOptimization
from utils import logMap, quaternion_to_rotation_matrix

# Global variables
robot = "go2"
json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'config')

# Load the dataset
if not os.path.exists(f"{robot}_test.npz"):
    with zipfile.ZipFile(f"{robot}_test.zip", 'r') as zip_ref:
        zip_ref.extractall()
dataset = np.load(f"{robot}_test.npz", allow_pickle=True)

# Load the initial state from the dataset
joint_state = dataset["joint_states"][0]
base_state = dataset["base_states"][0]
contact_state = dataset["contact_states"][0]

# Load the ground truth base positions and orientations from the dataset
ground_truth_base_positions = [gt.position for gt in dataset["base_pose_ground_truth"]]
ground_truth_base_orientations = [gt.orientation for gt in dataset["base_pose_ground_truth"]]

def default_config():
    # Read the json file
    json_file = f'{json_path}/{robot}_pytest.json'
    with open(json_file, 'r') as f:
        config = json.load(f)

    params = {
        'x0': config['imu_angular_velocity_covariance'][0],
        'y0': config['imu_angular_velocity_covariance'][1],
        'z0': config['imu_angular_velocity_covariance'][2],
        'x1': config['imu_linear_acceleration_covariance'][0],
        'y1': config['imu_linear_acceleration_covariance'][1],
        'z1': config['imu_linear_acceleration_covariance'][2],
        'jv': config['joint_position_variance'],
    }
    print(f"Default params: {params}")
    return params

# 0. Define the function you want to optimize (the "Black Box")
def fn(x0, y0, z0, x1, y1, z1, jv):
    # Read the json file
    json_file = f'{json_path}/{robot}_pytest.json'
    with open(json_file, 'r') as f:
        config = json.load(f)

    # Replace the parameters with the new values
    config['imu_angular_velocity_covariance'][0] = x0
    config['imu_angular_velocity_covariance'][1] = y0
    config['imu_angular_velocity_covariance'][2] = z0
    config['imu_linear_acceleration_covariance'][0] = x1
    config['imu_linear_acceleration_covariance'][1] = y1
    config['imu_linear_acceleration_covariance'][2] = z1
    config['joint_position_variance'] = jv

    # Write the config to a temporary file
    temp_file = f'{json_path}/{robot}_temp.json'
    with open(temp_file, 'w') as f:
        json.dump(config, f)

    # Initialize the Serow framework
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}_temp.json")
    initial_state = serow_framework.get_state(allow_invalid=True)
    initial_state.set_joint_state(joint_state)
    initial_state.set_base_state(base_state)
    initial_state.set_contact_state(contact_state)
    serow_framework.set_state(initial_state)
     
    # Run filter
    position_error = 0.0
    orientation_error = 0.0
    steps = 0
    successful_steps = 0
    for imu, joint, ft in zip(dataset["imu"], dataset["joints"], dataset["ft"]):
        status = serow_framework.filter(imu, joint, ft, None)
        if status:
            state = serow_framework.get_state(allow_invalid=True)
            base_position = np.array(state.get_base_position())
            base_orientation = np.array(state.get_base_orientation())
            gt_base_position = np.array(ground_truth_base_positions[steps])
            gt_base_orientation = np.array(ground_truth_base_orientations[steps])
            position_error += np.linalg.norm(base_position - gt_base_position, axis=0)
            orientation_error += np.linalg.norm(logMap(
                quaternion_to_rotation_matrix(gt_base_orientation).transpose()
                @ quaternion_to_rotation_matrix(base_orientation),
            ), axis=0)
            successful_steps += 1
        steps += 1
 
    # Normalize errors by number of successful steps to avoid bias toward longer sequences
    if successful_steps == 0:
        return -1e10  # Return poor score if no successful steps
    
    # Normalize and combine errors 
    normalized_position_error = position_error / successful_steps
    normalized_orientation_error = orientation_error / successful_steps
    
    # Return negative sum for maximization (BayesianOptimization maximizes)
    return -(normalized_position_error + normalized_orientation_error)


if __name__ == "__main__":
    # 1. Get the default parameters
    default_params = default_config()

    # 2. Define the parameter bounds (the search space)
    pbounds = {'x0': (1e-6, 1e-4), 
               'y0': (1e-6, 1e-4), 
               'z0': (1e-6, 1e-4), 
               'x1': (1e-4, 1e-2), 
               'y1': (1e-4, 1e-2), 
               'z1': (1e-4, 1e-2),
               'jv' : (1e-5, 1e-3)}

    # 3. Initialize the optimizer
    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=42,
    )

    # Load the optimizer state from the file (if it exists)
    if os.path.exists(f"{robot}_optimizer_state.json"):
        state_file = f"{robot}_optimizer_state.json"
        print(f"Loading optimizer state from {state_file}")
        optimizer.load_state(state_file)
    else:
        # Probe the default parameters (once at the beginning to guide the optimizer)
        optimizer.probe(params=default_params, lazy=True)

    # 4. Perform the optimization
    # init_points: How many steps of random exploration you want to perform
    # n_iter: How many steps of bayesian optimization you want to perform
    optimizer.maximize(
        n_iter=500,
    )

    # 5. Get the best results
    print(f"Best optimization parameters: {optimizer.max['params']}")
    print(f"Best optimization function value: {optimizer.max['target']}")

    # 6. Save the optimizer state to the file
    optimizer.save_state(f"{robot}_optimizer_state.json")

    # 7. Remove the temporary file
    os.remove(f"{json_path}/{robot}_temp.json")
