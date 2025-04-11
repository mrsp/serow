#!/usr/bin/env python3

import numpy as np
from serow import ContactEKF, BaseState
from serow import ImuMeasurement, KinematicMeasurement, OdometryMeasurement
from ddpg import DDPG
from read_mcap import read_initial_base_state, read_kinematic_measurements, read_imu_measurements
import matplotlib.pyplot as plt


def main():
    # Read the measurement mcap file
    kinematic_measurements = read_kinematic_measurements("/tmp/serow_measurements.mcap")
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    initial_state = read_initial_base_state("/tmp/serow_proprioception.mcap")

    # Get the contacts frame
    contacts_frame = set(initial_state.contacts_position.keys())
    print(f"Contacts frame: {contacts_frame}")
    
    # Initialize the EKF
    ekf = ContactEKF()
    
    # Create initial state
    state = initial_state

    # Initialize the EKF
    point_feet = True  # Assuming point feet or flat feet
    g = 9.81  # Gravity constant
    imu_rate = 500.0  # IMU update rate in Hz
    outlier_detection = False  # Enable outlier detection
    
    ekf.init(state, contacts_frame, point_feet, g, imu_rate, outlier_detection)
    
    # Run a few prediction/update steps
    step = 0
    for imu, kin in zip(imu_measurements, kinematic_measurements):
        # Set action
        ekf.set_action(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        # Predict step
        ekf.predict(state, imu, kin)
        
        # Update step (pass None for both optional parameters)
        ekf.update(state, kin, None, None)
        
        # Print some state information
        print(f"\nStep {step}:")
        print(f"Position: {state.base_position}")
        print(f"Velocity: {state.base_linear_velocity}")
        print(f"Orientation: {state.base_orientation}")

        # Print contact positions and orientations
        print("\nContact Positions:")
        for frame_name, position in state.contacts_position.items():
            print(f"{frame_name}: {position}")
        
        if state.contacts_orientation:
            print("\nContact Orientations:")
            for frame_name, orientation in state.contacts_orientation.items():
                print(f"{frame_name}: {orientation}")

        step += 1

if __name__ == "__main__":
    main() 
