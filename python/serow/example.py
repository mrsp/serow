#!/usr/bin/env python3

import numpy as np
from serow import ContactEKF, BaseState
from serow import ImuMeasurement, KinematicMeasurement, OdometryMeasurement

def main():
    # Initialize the EKF
    ekf = ContactEKF()
    
    # Create initial state
    state = BaseState()
    state.timestamp = 0.0
    state.base_position = np.array([0.0, 0.0, 1.0])  # Start 1m above ground
    state.base_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    state.base_linear_velocity = np.array([0.0, 0.0, 0.0])
    state.imu_angular_velocity_bias = np.array([0.0, 0.0, 0.0])
    state.imu_linear_acceleration_bias = np.array([0.0, 0.0, 0.0])
    
    # Initialize contact positions (example for a bipedal robot)
    state.contacts_position = {
        "left_foot": np.array([0.0, 0.1, 0.0]),  # 10cm to the left
        "right_foot": np.array([0.0, -0.1, 0.0])  # 10cm to the right
    }
    
    # Initialize contact orientations
    state.contacts_orientation = {
        "left_foot": np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        "right_foot": np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    }
    
    # Initialize covariances
    state.base_position_cov = np.eye(3) * 0.01  # 10cm uncertainty
    state.base_orientation_cov = np.eye(3) * 0.01  # ~5.7 degrees uncertainty
    state.base_linear_velocity_cov = np.eye(3) * 0.1  # 0.3 m/s uncertainty
    state.imu_angular_velocity_bias_cov = np.eye(3) * 0.0001
    state.imu_linear_acceleration_bias_cov = np.eye(3) * 0.0001
    state.contacts_position_cov = {
        "left_foot": np.eye(3) * 0.0001,
        "right_foot": np.eye(3) * 0.0001
    }
    state.contacts_orientation_cov = {
        "left_foot": np.eye(3) * 0.0001,
        "right_foot": np.eye(3) * 0.0001
    }
    
    # Initialize the EKF
    contacts_frame = {"left_foot", "right_foot"}
    point_feet = True  # Assuming point feet
    g = 9.81  # Gravity constant
    imu_rate = 1000.0  # IMU update rate in Hz
    outlier_detection = True  # Enable outlier detection
    
    ekf.init(state, contacts_frame, point_feet, g, imu_rate, outlier_detection)
    
    # Create IMU measurement
    imu = ImuMeasurement()
    imu.timestamp = 0.001  # 1ms after initialization
    imu.angular_velocity = np.array([0.0, 0.0, 0.0])  # No rotation
    imu.linear_acceleration = np.array([0.0, 0.0, -g])  # Gravity only
    imu.angular_velocity_cov = np.eye(3) * 0.0001
    imu.linear_acceleration_cov = np.eye(3) * 0.0001
    imu.angular_velocity_bias_cov = np.eye(3) * 0.000001
    imu.linear_acceleration_bias_cov = np.eye(3) * 0.000001
    
    # Create kinematic measurement
    kin = KinematicMeasurement()
    kin.timestamp = 0.001
    kin.contacts_status = {
        "left_foot": True,  # Left foot in contact
        "right_foot": True  # Right foot in contact
    }
    kin.contacts_probability = {
        "left_foot": 1.0,
        "right_foot": 1.0
    }
    kin.contacts_position = {
        "left_foot": np.array([0.0, 0.1, 0.0]),
        "right_foot": np.array([0.0, -0.1, 0.0])
    }
    kin.contacts_orientation = {
        "left_foot": np.array([1.0, 0.0, 0.0, 0.0]),
        "right_foot": np.array([1.0, 0.0, 0.0, 0.0])
    }
    kin.position_slip_cov = np.eye(3) * 0.0001
    kin.orientation_slip_cov = np.eye(3) * 0.0001
    kin.contacts_position_noise = {
        "left_foot": np.eye(3) * 0.0001,
        "right_foot": np.eye(3) * 0.0001
    }
    kin.contacts_orientation_noise = {
        "left_foot": np.eye(3) * 0.0001,
        "right_foot": np.eye(3) * 0.0001
    }
    
    # Create odometry measurement (optional)
    odom = OdometryMeasurement()
    odom.timestamp = 0.001
    odom.base_position = np.array([0.0, 0.0, 1.0])
    odom.base_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    odom.base_position_cov = np.eye(3) * 0.01
    odom.base_orientation_cov = np.eye(3) * 0.01
    
    # Run a few prediction/update steps
    for i in range(10):
        # Update timestamps
        dt = 0.001  # 1ms time stepS
        imu.timestamp += dt
        kin.timestamp += dt
        odom.timestamp += dt
        
        # Set action
        for cf in contacts_frame:
            ekf.set_action(cf, np.array([1.0, 1.0]))

        # Predict step
        ekf.predict(state, imu, kin)
        
        # Update step (pass None for both optional parameters)
        ekf.update(state, kin, None, None)
        
        for cf in contacts_frame:
            innovation = np.zeros(3)
            covariance = np.zeros((3, 3))
            success, innovation, covariance = ekf.get_contact_position_innovation(cf)
            if success:
                print(f"Contact frame: {cf}")
                print(f"Innovation: {innovation}")
                print(f"Covariance: {covariance}")

        # Print some state information
        print(f"\nStep {i+1}:")
        print(f"Position: {state.base_position}")
        print(f"Velocity: {state.base_linear_velocity}")
        print(f"Orientation: {state.base_orientation}")
        print(f"Left foot position: {state.contacts_position['left_foot']}")
        print(f"Right foot position: {state.contacts_position['right_foot']}")

if __name__ == "__main__":
    main() 
