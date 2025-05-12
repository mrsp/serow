import mcap
from mcap.reader import make_reader
import numpy as np
import serow
import sys
import os
import matplotlib.pyplot as plt

USE_GROUND_TRUTH = True

# Add the build directory to Python path to find generated schemas
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # This points to serow directory
build_dir = os.path.join(project_root, 'build', 'generated')
print(f"Looking for schemas in: {build_dir}")  # Debug print

if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
else:
    raise ImportError(f"Could not find generated schemas in {build_dir}. Please build the project with Python code generation enabled:\n"
                     f"1. cd {project_root}\n"
                     f"2. mkdir -p build\n"
                     f"3. cd build\n"
                     f"4. cmake -DGENERATE_PYTHON_SCHEMAS=ON ..\n"
                     f"5. make")

try:
    from foxglove.Vector3 import Vector3
    from foxglove.Quaternion import Quaternion
    from foxglove.Matrix3 import Matrix3
    from foxglove.Time import Time
    from foxglove.BaseState import BaseState as FbBaseState
    from foxglove.KinematicMeasurement import KinematicMeasurement as FbKinematicMeasurement
    from foxglove.ImuMeasurement import ImuMeasurement as FbImuMeasurement
    from foxglove.FrameTransform import FrameTransform as FbFrameTransform  
    from foxglove.JointMeasurements import JointMeasurements as FbJointMeasurements
    from foxglove.ForceTorqueMeasurements import ForceTorqueMeasurements as FbForceTorqueMeasurements
    from foxglove.ContactState import ContactState as FbContactState
    from foxglove.JointState import JointState  as FbJointState
except ImportError as e:
    raise ImportError(f"Failed to import FlatBuffer schemas. Please ensure the project is built with Python code generation enabled. Error: {e}")

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters: 
    R : numpy array with shape (3, 3)
        The rotation matrix
        
    Returns:
    numpy array with shape (4,)
        The corresponding quaternion
    """
    # Compute the trace of the matrix
    trace = np.trace(R)
    q = np.array([1.0, 0.0, 0.0, 0.0])

    # Check if the matrix is close to a pure rotation matrix
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2.0  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
        q = np.array([qw, qx, qy, qz])   
    else:
        # Compute the largest diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # S=4*qx
            qx = 0.25 * S
            qy = (R[1, 0] + R[0, 1]) / S
            qz = (R[2, 0] + R[0, 2]) / S
            qw = (R[2, 1] - R[1, 2]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # S=4*qy  
            qy = 0.25 * S       
            qx = (R[1, 0] + R[0, 1]) / S
            qz = (R[2, 1] + R[1, 2]) / S
            qw = (R[0, 2] - R[2, 0]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # S=4*qz  
            qz = 0.25 * S
            qx = (R[2, 0] + R[0, 2]) / S
            qy = (R[2, 1] + R[1, 2]) / S
            qw = (R[1, 0] - R[0, 1]) / S
        q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    
    Parameters:
    q : numpy array with shape (4,)
        The quaternion in the form [w, x, y, z]
        
    Returns:
    numpy array with shape (3, 3)
        The corresponding rotation matrix
    """
    # Ensure q is normalized
    q = q / np.linalg.norm(q)
    
    # Extract the values from q
    w, x, y, z = q
    
    # Compute the rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def logMap(R):
    R11 = R[0, 0];
    R12 = R[0, 1];
    R13 = R[0, 2];
    R21 = R[1, 0];
    R22 = R[1, 1];
    R23 = R[1, 2];
    R31 = R[2, 0];
    R32 = R[2, 1];
    R33 = R[2, 2];

    trace = R.trace();

    omega = np.zeros(3)

    # Special case when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    if (trace + 1.0 < 1e-3) :
        if (R33 > R22 and R33 > R11) :
            # R33 is the largest diagonal, a=3, b=1, c=2
            W = R21 - R12;
            Q1 = 2.0 + 2.0 * R33;
            Q2 = R31 + R13
            Q3 = R23 + R32
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W)
            sgn_w = -1.0 if W < 0 else 1.0
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q2, Q3, Q1])
        elif (R22 > R11):
            # R22 is the largest diagonal, a=2, b=3, c=1
            W = R13 - R31;
            Q1 = 2.0 + 2.0 * R22;
            Q2 = R23 + R32;
            Q3 = R12 + R21;
            r = np.sqrt(Q1);
            one_over_r = 1 / r;
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            sgn_w = -1.0 if W < 0 else 1.0;
            mag = np.pi - (2 * sgn_w * W) / norm;
            scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * np.array([Q3, Q1, Q2]);
        else:
            # R11 is the largest diagonal, a=1, b=2, c=3
            W = R32 - R23;
            Q1 = 2.0 + 2.0 * R11;
            Q2 = R12 + R21;
            Q3 = R31 + R13;
            r = np.sqrt(Q1);
            one_over_r = 1 / r;
            norm = np.sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            sgn_w = -1.0 if W < 0 else 1.0;
            mag = np.pi - (2 * sgn_w * W) / norm;
            scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * np.array([Q1, Q2, Q3])
    else:
        magnitude = 0.0;
        tr_3 = trace - 3.0;  # could be non-negative if the matrix is off orthogonal
        if (tr_3 < -1e-6):
            # this is the normal case -1 < trace < 3
            theta = np.arccos((trace - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            # see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0;

        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12]);
    return omega;

def decode_imu_measurement(data: bytes) -> serow.ImuMeasurement:
    """Decode a FlatBuffer message into an ImuMeasurement object."""
    fb_msg = FbImuMeasurement.GetRootAsImuMeasurement(data, 0)
    msg = serow.ImuMeasurement()
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    if timestamp:
        msg.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode linear acceleration
    if fb_msg.LinearAcceleration():
        msg.linear_acceleration = np.array([
            fb_msg.LinearAcceleration().X(),
            fb_msg.LinearAcceleration().Y(),
            fb_msg.LinearAcceleration().Z()
        ])
    
    # Decode angular velocity
    if fb_msg.AngularVelocity():
        msg.angular_velocity = np.array([
            fb_msg.AngularVelocity().X(),
            fb_msg.AngularVelocity().Y(),
            fb_msg.AngularVelocity().Z()
        ])
    
    # Decode orientation
    if fb_msg.Orientation():
        msg.orientation = np.array([
            fb_msg.Orientation().W(),
            fb_msg.Orientation().X(),
            fb_msg.Orientation().Y(),
            fb_msg.Orientation().Z()
        ])
    
    # Decode linear acceleration covariance
    if fb_msg.LinearAccelerationCov():
        matrix = fb_msg.LinearAccelerationCov()
        msg.linear_acceleration_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode angular velocity covariance
    if fb_msg.AngularVelocityCov():
        matrix = fb_msg.AngularVelocityCov()
        msg.angular_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode angular velocity bias covariance
    if fb_msg.AngularVelocityBiasCov():
        matrix = fb_msg.AngularVelocityBiasCov()
        msg.angular_velocity_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode linear acceleration bias covariance
    if fb_msg.LinearAccelerationBiasCov():
        matrix = fb_msg.LinearAccelerationBiasCov()
        msg.linear_acceleration_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode angular acceleration
    if fb_msg.AngularAcceleration():
        msg.angular_acceleration = np.array([
            fb_msg.AngularAcceleration().X(),
            fb_msg.AngularAcceleration().Y(),
            fb_msg.AngularAcceleration().Z()
        ])
    
    return msg

def decode_kinematic_measurement(data: bytes) -> serow.KinematicMeasurement:
    """Decode a FlatBuffer message into a KinematicMeasurement object."""
    fb_msg = FbKinematicMeasurement.GetRootAsKinematicMeasurement(data, 0)
    msg = serow.KinematicMeasurement()
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    if timestamp:
        msg.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode base linear velocity
    if fb_msg.BaseLinearVelocity():
        msg.base_linear_velocity = np.array([
            fb_msg.BaseLinearVelocity().X(),
            fb_msg.BaseLinearVelocity().Y(),
            fb_msg.BaseLinearVelocity().Z()
        ])
    
    # Decode base orientation
    if fb_msg.BaseOrientation():
        msg.base_orientation = np.array([
            fb_msg.BaseOrientation().W(),
            fb_msg.BaseOrientation().X(),
            fb_msg.BaseOrientation().Y(),
            fb_msg.BaseOrientation().Z()
        ])
    
    # Decode contact names and status
    contacts_status = {}
    contacts_probability = {}
    contacts_position = {}
    base_to_foot_positions = {}
    contacts_position_noise = {}
    contacts_orientation = {}
    contacts_orientation_noise = {}

    for i in range(fb_msg.ContactNamesLength()):
        name = fb_msg.ContactNames(i).decode()
        if i < fb_msg.ContactsStatusLength():
            contacts_status[name] = fb_msg.ContactsStatus(i)
        if i < fb_msg.ContactsProbabilityLength():
            contacts_probability[name] = fb_msg.ContactsProbability(i)
        if i < fb_msg.ContactsPositionLength():
            pos = fb_msg.ContactsPosition(i)
            contacts_position[name] = np.array([pos.X(), pos.Y(), pos.Z()])
        if i < fb_msg.BaseToFootPositionsLength():
            pos = fb_msg.BaseToFootPositions(i)
            base_to_foot_positions[name] = np.array([pos.X(), pos.Y(), pos.Z()])
        if i < fb_msg.ContactsPositionNoiseLength():
            matrix = fb_msg.ContactsPositionNoise(i)
            contacts_position_noise[name] = np.array([
                [matrix.M00(), matrix.M01(), matrix.M02()],
                [matrix.M10(), matrix.M11(), matrix.M12()],
                [matrix.M20(), matrix.M21(), matrix.M22()]
            ])
        if i < fb_msg.ContactsOrientationLength():
            quat = fb_msg.ContactsOrientation(i)
            contacts_orientation[name] = np.array([
                quat.W(), quat.X(), quat.Y(), quat.Z()
            ])
        if i < fb_msg.ContactsOrientationNoiseLength():
            matrix = fb_msg.ContactsOrientationNoise(i)
            contacts_orientation_noise[name] = np.array([
                [matrix.M00(), matrix.M01(), matrix.M02()],
                [matrix.M10(), matrix.M11(), matrix.M12()],
                [matrix.M20(), matrix.M21(), matrix.M22()]
            ])
    
    msg.contacts_status = contacts_status
    msg.contacts_probability = contacts_probability
    msg.contacts_position = contacts_position
    msg.base_to_foot_positions = base_to_foot_positions
    msg.contacts_position_noise = contacts_position_noise
    msg.contacts_orientation = contacts_orientation
    msg.contacts_orientation_noise = contacts_orientation_noise

    # Decode COM measurements
    if fb_msg.ComAngularMomentumDerivative():
       msg.com_angular_momentum_derivative = np.array([
            fb_msg.ComAngularMomentumDerivative().X(),
            fb_msg.ComAngularMomentumDerivative().Y(),
            fb_msg.ComAngularMomentumDerivative().Z()
        ])
    
    if fb_msg.ComPosition():
        msg.com_position = np.array([
            fb_msg.ComPosition().X(),
            fb_msg.ComPosition().Y(),
            fb_msg.ComPosition().Z()
        ])
    
    if fb_msg.ComLinearAcceleration():
        msg.com_linear_acceleration = np.array([
            fb_msg.ComLinearAcceleration().X(),
            fb_msg.ComLinearAcceleration().Y(),
            fb_msg.ComLinearAcceleration().Z()
        ])
    
    # Decode covariance matrices
    if fb_msg.BaseLinearVelocityCov():
        matrix = fb_msg.BaseLinearVelocityCov()
        msg.base_linear_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseOrientationCov():
        matrix = fb_msg.BaseOrientationCov()
        msg.base_orientation_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.PositionSlipCov():
        matrix = fb_msg.PositionSlipCov()
        msg.position_slip_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.OrientationSlipCov():
        matrix = fb_msg.OrientationSlipCov()
        msg.orientation_slip_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.PositionCov():
        matrix = fb_msg.PositionCov()
        msg.position_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.OrientationCov():
        matrix = fb_msg.OrientationCov()
        msg.orientation_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComPositionProcessCov():
        matrix = fb_msg.ComPositionProcessCov()
        msg.com_position_process_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComLinearVelocityProcessCov():
        matrix = fb_msg.ComLinearVelocityProcessCov()
        msg.com_linear_velocity_process_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ExternalForcesProcessCov():
        matrix = fb_msg.ExternalForcesProcessCov()
        msg.external_forces_process_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComPositionCov():
        matrix = fb_msg.ComPositionCov()
        msg.com_position_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComLinearAccelerationCov():
        matrix = fb_msg.ComLinearAccelerationCov()
        msg.com_linear_acceleration_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    return msg

def decode_base_state(data: bytes) -> serow.BaseState:
    """Decode a FlatBuffer message into an BaseState object."""
    fb_msg = FbBaseState.GetRootAsBaseState(data, 0)
    msg = serow.BaseState()
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    if timestamp:
        msg.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode base position
    if fb_msg.BasePosition():
        msg.base_position = np.array([
            fb_msg.BasePosition().X(),
            fb_msg.BasePosition().Y(),
            fb_msg.BasePosition().Z()
        ])
    
    # Decode base orientation
    if fb_msg.BaseOrientation():
        msg.base_orientation = np.array([
            fb_msg.BaseOrientation().W(),
            fb_msg.BaseOrientation().X(),
            fb_msg.BaseOrientation().Y(),
            fb_msg.BaseOrientation().Z()
        ])
    
    # Decode base linear velocity
    if fb_msg.BaseLinearVelocity():
        msg.base_linear_velocity = np.array([
            fb_msg.BaseLinearVelocity().X(),
            fb_msg.BaseLinearVelocity().Y(),
            fb_msg.BaseLinearVelocity().Z()
        ])
    
    # Decode base angular velocity
    if fb_msg.BaseAngularVelocity():
        msg.base_angular_velocity = np.array([
            fb_msg.BaseAngularVelocity().X(),
            fb_msg.BaseAngularVelocity().Y(),
            fb_msg.BaseAngularVelocity().Z()
        ])
    
    # Decode base linear acceleration
    if fb_msg.BaseLinearAcceleration():
        msg.base_linear_acceleration = np.array([
            fb_msg.BaseLinearAcceleration().X(),
            fb_msg.BaseLinearAcceleration().Y(),
            fb_msg.BaseLinearAcceleration().Z()
        ])
    
    # Decode base angular acceleration
    if fb_msg.BaseAngularAcceleration():
        msg.base_angular_acceleration = np.array([
            fb_msg.BaseAngularAcceleration().X(),
            fb_msg.BaseAngularAcceleration().Y(),
            fb_msg.BaseAngularAcceleration().Z()
        ])
    
    # Decode IMU linear acceleration bias
    if fb_msg.ImuLinearAccelerationBias():
        msg.imu_linear_acceleration_bias = np.array([
            fb_msg.ImuLinearAccelerationBias().X(),
            fb_msg.ImuLinearAccelerationBias().Y(),
            fb_msg.ImuLinearAccelerationBias().Z()
        ])
    
    # Decode IMU angular velocity bias
    if fb_msg.ImuAngularVelocityBias():
        msg.imu_angular_velocity_bias = np.array([
            fb_msg.ImuAngularVelocityBias().X(),
            fb_msg.ImuAngularVelocityBias().Y(),
            fb_msg.ImuAngularVelocityBias().Z()
        ])
    
    # Decode covariance matrices
    if fb_msg.BasePositionCov():
        matrix = fb_msg.BasePositionCov()
        msg.base_position_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseOrientationCov():
        matrix = fb_msg.BaseOrientationCov()
        msg.base_orientation_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseLinearVelocityCov():
        matrix = fb_msg.BaseLinearVelocityCov()
        msg.base_linear_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseAngularVelocityCov():
        matrix = fb_msg.BaseAngularVelocityCov()
        msg.base_angular_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ImuLinearAccelerationBiasCov():
        matrix = fb_msg.ImuLinearAccelerationBiasCov()
        msg.imu_linear_acceleration_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ImuAngularVelocityBiasCov():
        matrix = fb_msg.ImuAngularVelocityBiasCov()
        msg.imu_angular_velocity_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode contact positions
    contacts_position = {}
    for i in range(fb_msg.ContactsPositionLength()):
        pos = fb_msg.ContactsPosition(i)
        contacts_position[fb_msg.ContactNames(i).decode()] = np.array([
            pos.X(), pos.Y(), pos.Z()
        ])
    
    # Decode contact orientations
    contacts_orientation = {}
    for i in range(fb_msg.ContactsOrientationLength()):
        quat = fb_msg.ContactsOrientation(i)
        contacts_orientation[fb_msg.ContactNames(i).decode()] = np.array([
            quat.W(), quat.X(), quat.Y(), quat.Z()
        ])
    
    # Decode contact position covariances
    contacts_position_cov = {}
    for i in range(fb_msg.ContactsPositionCovLength()):
        matrix = fb_msg.ContactsPositionCov(i)
        contacts_position_cov[fb_msg.ContactNames(i).decode()] = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode contact orientation covariances
    contacts_orientation_cov = {}
    for i in range(fb_msg.ContactsOrientationCovLength()):
        matrix = fb_msg.ContactsOrientationCov(i)
        contacts_orientation_cov[fb_msg.ContactNames(i).decode()] = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    msg.contacts_position = contacts_position
    msg.contacts_orientation = contacts_orientation
    msg.contacts_position_cov = contacts_position_cov
    msg.contacts_orientation_cov = contacts_orientation_cov

    # Decode feet positions
    feet_position = {}
    for i in range(fb_msg.FeetPositionLength()):
        pos = fb_msg.FeetPosition(i)
        feet_position[fb_msg.ContactNames(i).decode()] = np.array([
            pos.X(), pos.Y(), pos.Z()
        ])
    
    # Decode feet orientations
    feet_orientation = {}
    for i in range(fb_msg.FeetOrientationLength()):
        quat = fb_msg.FeetOrientation(i)
        feet_orientation[fb_msg.ContactNames(i).decode()] = np.array([
            quat.W(), quat.X(), quat.Y(), quat.Z()
        ])
    
    # Decode feet linear velocities
    feet_linear_velocity = {}
    for i in range(fb_msg.FeetLinearVelocityLength()):
        vel = fb_msg.FeetLinearVelocity(i)
        feet_linear_velocity[fb_msg.ContactNames(i).decode()] = np.array([
            vel.X(), vel.Y(), vel.Z()
        ])
    
    # Decode feet angular velocities
    feet_angular_velocity = {}
    for i in range(fb_msg.FeetAngularVelocityLength()):
        vel = fb_msg.FeetAngularVelocity(i)
        feet_angular_velocity[fb_msg.ContactNames(i).decode()] = np.array([
            vel.X(), vel.Y(), vel.Z()
        ])
    
    msg.feet_position = feet_position
    msg.feet_orientation = feet_orientation
    msg.feet_linear_velocity = feet_linear_velocity
    msg.feet_angular_velocity = feet_angular_velocity

    return msg

def decode_contact_state(data: bytes) -> serow.ContactState:
    """Decode a FlatBuffer message into an ContactState object."""
    fb_msg = FbContactState.GetRootAsContactState(data, 0)
    msg = serow.ContactState()
    
    # Decode timestamp  
    timestamp = fb_msg.Timestamp()
    if timestamp:
        msg.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode contact statuses and other fields
    contacts_status = {}
    contacts_probability = {}
    contacts_force = {}
    contacts_torque = {}
    
    for i in range(fb_msg.ContactsLength()):
        contact = fb_msg.Contacts(i)
        name = fb_msg.ContactNames(i).decode()
        
        contacts_status[name] = contact.Status()
        contacts_probability[name] = contact.Probability()
        
        # Decode force
        force = contact.Force()
        if force:
            contacts_force[name] = np.array([force.X(), force.Y(), force.Z()])
            
        # Decode torque if available
        torque = contact.Torque()
        if torque:
            if name not in contacts_torque:
                contacts_torque[name] = np.array([torque.X(), torque.Y(), torque.Z()])
    
    msg.contacts_status = contacts_status
    msg.contacts_probability = contacts_probability
    msg.contacts_force = contacts_force
    if contacts_torque:
        msg.contacts_torque = contacts_torque

    return msg

def decode_base_pose_ground_truth(data: bytes) -> serow.BasePoseGroundTruth:
    """Decode a FlatBuffer message into an BasePoseGroundTruth object."""
    fb_msg = FbFrameTransform.GetRootAsFrameTransform(data, 0)
    msg = serow.BasePoseGroundTruth()
    
    # Decode timestamp
    if fb_msg.Timestamp():
        msg.timestamp = fb_msg.Timestamp().Sec() + fb_msg.Timestamp().Nsec() * 1e-9
    
    # Decode position
    if fb_msg.Translation():   
        msg.position = np.array([
            fb_msg.Translation().X(),
            fb_msg.Translation().Y(),
            fb_msg.Translation().Z()
        ])              
    
    # Decode orientation
    if fb_msg.Rotation():
        msg.orientation = np.array([
            fb_msg.Rotation().W(),
            fb_msg.Rotation().X(),
            fb_msg.Rotation().Y(),
            fb_msg.Rotation().Z()
        ])
    
    return msg

def decode_joint_measurement(data: bytes) -> dict[str, serow.JointMeasurement]:
    """Decode a FlatBuffer message into a dictionary of JointMeasurement objects."""
    fb_msg = FbJointMeasurements.GetRootAsJointMeasurements(data, 0)
    joint_measurements = {}
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    timestamp_sec = 0.0
    if timestamp:
        timestamp_sec = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode joint names and positions
    for i in range(fb_msg.NamesLength()):
        joint_name = fb_msg.Names(i).decode()
        msg = serow.JointMeasurement()
        msg.timestamp = timestamp_sec
        
        if i < fb_msg.PositionsLength():
            msg.position = fb_msg.Positions(i)
        if i < fb_msg.VelocitiesLength():
            msg.velocity = fb_msg.Velocities(i)
        
        joint_measurements[joint_name] = msg
    
    return joint_measurements

def decode_force_torque_measurement(data: bytes) -> dict[str, serow.ForceTorqueMeasurement]:
    """Decode a FlatBuffer message into a dictionary of ForceTorqueMeasurement objects."""
    fb_msg = FbForceTorqueMeasurements.GetRootAsForceTorqueMeasurements(data, 0)
    ft_measurements = {}
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    timestamp_sec = 0.0
    if timestamp:
        timestamp_sec = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode frame names and force/torque values
    for i in range(fb_msg.FrameNamesLength()):
        frame_name = fb_msg.FrameNames(i).decode()
        msg = serow.ForceTorqueMeasurement()
        msg.timestamp = timestamp_sec
        
        if i < fb_msg.ForcesLength():
            force = fb_msg.Forces(i)
            msg.force = np.array([force.X(), force.Y(), force.Z()])
        
        if i < fb_msg.TorquesLength():
            torque = fb_msg.Torques(i)
            msg.torque = np.array([torque.X(), torque.Y(), torque.Z()])
        
        ft_measurements[frame_name] = msg
    
    return ft_measurements

def decode_joint_state(data: bytes) -> serow.JointState:
    """Decode a FlatBuffer message into a JointState object."""
    fb_msg = FbJointState.GetRootAs(data, 0)
    joint_state = serow.JointState()
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    if timestamp:
        joint_state.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Get the arrays from the FlatBuffer message
    names = []
    positions = []
    velocities = []
    
    for i in range(fb_msg.NamesLength()):
        name = fb_msg.Names(i)
        if name:
            names.append(name.decode())
    
    for i in range(fb_msg.PositionsLength()):
        pos = fb_msg.Positions(i)
        positions.append(pos)
    
    for i in range(fb_msg.VelocitiesLength()):
        vel = fb_msg.Velocities(i)
        velocities.append(vel)
    
    # Create new dictionaries
    joints_position = {}
    joints_velocity = {}
    
    # Fill the dictionaries
    for i, name in enumerate(names):
        if i < len(positions):
            joints_position[name] = positions[i]
        if i < len(velocities):
            joints_velocity[name] = velocities[i]
    
    # Assign the dictionaries to the joint_state object
    joint_state.joints_position = joints_position
    joint_state.joints_velocity = joints_velocity
    return joint_state

def read_joint_states(file_path: str):
    """Read and decode joint state messages from an MCAP file."""
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            joint_states_list = []
            
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/joint_state":
                    joint_state = decode_joint_state(message.data)
                    joint_states_list.append(joint_state)
            
            print(f"Found {len(joint_states_list)} joint states")
            return joint_states_list
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def read_imu_measurements(file_path: str):
    """Read and decode messages from an MCAP file."""
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            imu_measurements = []
            
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/imu":
                    msg = decode_imu_measurement(message.data)
                    imu_measurements.append(msg)
            
            print(f"Found {len(imu_measurements)} IMU measurements")
            return imu_measurements
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def read_kinematic_measurements(file_path: str):
    """Read and decode messages from an MCAP file."""
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            kinematic_measurements = []
            
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/kin":
                    msg = decode_kinematic_measurement(message.data)
                    kinematic_measurements.append(msg)
            
            print(f"Found {len(kinematic_measurements)} kinematic measurements")
            return kinematic_measurements
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def read_base_states(file_path: str):
    """Read and decode the base states from an MCAP file.
    
    Args:
        file_path: Path to the MCAP file
        
    Returns:
        BaseState: The decoded base states, or None if not found
    """
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            base_states = []    
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/base_state":
                    base_states.append(decode_base_state(message.data))
        
            print(f"Found {len(base_states)} base states")
            return base_states
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def read_contact_states(file_path: str):
    """Read and decode the contact states from an MCAP file.
    
    Args:
        file_path: Path to the MCAP file
        
    Returns:
        ContactState: The decoded contact states, or None if not found
    """
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            contact_states = []
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/contact_state":
                    msg = decode_contact_state(message.data)
                    contact_states.append(msg)
            print(f"Found {len(contact_states)} contact states")
            return contact_states   
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def read_base_pose_ground_truth(file_path: str):
    """Read and decode the base pose ground truth from an MCAP file.
    
    Args:
        file_path: Path to the MCAP file
        
    Returns:
        BasePoseGroundTruth: The decoded base pose ground truth, or None if not found
    """
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            base_pose_ground_truth = []
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/base_pose_ground_truth":
                    msg = decode_base_pose_ground_truth(message.data)
                    base_pose_ground_truth.append(msg)
            
            print(f"Found {len(base_pose_ground_truth)} base pose ground truth")
            return base_pose_ground_truth
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def read_joint_measurements(file_path: str):
    """Read and decode joint measurement messages from an MCAP file."""
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            joint_measurements_list = []
            
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/joints":
                    joint_measurements = decode_joint_measurement(message.data)
                    joint_measurements_list.append(joint_measurements)
            
            print(f"Found {len(joint_measurements_list)} joint measurements")
            return joint_measurements_list
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def read_force_torque_measurements(file_path: str):
    """Read and decode force-torque measurement messages from an MCAP file."""
    try:
        with open(file_path, "rb") as f:
            reader = make_reader(f)
            ft_measurements_list = []
            
            for schema, channel, message in reader.iter_messages():
                if channel.topic == "/ft":
                    ft_measurements = decode_force_torque_measurement(message.data)
                    ft_measurements_list.append(ft_measurements)
            
            print(f"Found {len(ft_measurements_list)} force-torque measurements")
            return ft_measurements_list
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading MCAP file: {str(e)}")
        sys.exit(1)

def run_step(imu, joint, ft, gt, serow_framework, state, actions):
    # Set the actions
    for cf, action in actions.items():
        if (action is not None):
            serow_framework.set_action(cf, action)

    # Run the filter
    serow_framework.filter(imu, joint, ft, None, None)
    
    # Get the state
    state = serow_framework.get_state(allow_invalid=True)
        
    # Compute the reward
    rewards = {}
    done = 0.0
    for cf in state.get_contacts_frame():
        rewards[cf] = None
        
    for cf in state.get_contacts_frame():
        success = False
        innovation = np.zeros(3)
        base_position = np.zeros(3)
        base_orientation = np.zeros(4)
        covariance = np.zeros((3, 3))
        success, base_position, base_orientation, innovation, covariance = serow_framework.get_contact_position_innovation(cf)
        if success:
            # Check if innovation is too large or if covariance is not positive definite
            nis = innovation.dot(np.linalg.inv(covariance).dot(innovation))
            if nis > 1.0 or nis <= 0.0: 
                # filter diverged
                done = 1.0
                print(f"filter diverged")
                break
            contact_reward = -5e2 * nis
            rewards[cf] = contact_reward
            if USE_GROUND_TRUTH:
                position_reward = -1.0 * np.linalg.norm(base_position - gt.position)
                orientation_reward = -2.5 *np.linalg.norm(base_orientation - gt.orientation)
                # orientation_reward = -1e4 * np.linalg.norm(
                #     logMap(quaternion_to_rotation_matrix(gt.orientation).transpose() * quaternion_to_rotation_matrix(base_orientation)))
                rewards[cf] += position_reward + orientation_reward 
    return imu.timestamp, state, rewards, done

def sync_and_align_data(base_timestamps, base_position, base_orientation, gt_timestamps, gt_position, gt_orientation, align = False):
    # Find the common time range
    start_time = max(base_timestamps[0], gt_timestamps[0])
    end_time = min(base_timestamps[-1], gt_timestamps[-1])

    # Create a high-resolution common time grid
    common_timestamps = np.linspace(start_time, end_time, 1000)
    
    # Interpolate base position
    base_position_interp = np.zeros((len(common_timestamps), 3))
    for i in range(3):  # x, y, z
        base_position_interp[:, i] = np.interp(common_timestamps, base_timestamps, base_position[:, i])
    
    # Interpolate ground truth position
    gt_position_interp = np.zeros((len(common_timestamps), 3))
    for i in range(3):  # x, y, z
        gt_position_interp[:, i] = np.interp(common_timestamps, gt_timestamps, gt_position[:, i])

    # Interpolate orientations
    base_orientation_interp = np.zeros((len(common_timestamps), 4))
    gt_orientation_interp = np.zeros((len(common_timestamps), 4))
    for i in range(4):  # w, x, y, z
        base_orientation_interp[:, i] = np.interp(common_timestamps, base_timestamps, base_orientation[:, i])
        gt_orientation_interp[:, i] = np.interp(common_timestamps, gt_timestamps, gt_orientation[:, i])

    if align:
        # Compute the initial rigid body transformation from the first timestamp
        R_gt = quaternion_to_rotation_matrix(gt_orientation_interp[0])
        R_base = quaternion_to_rotation_matrix(base_orientation_interp[0])
        R = R_gt.transpose() @ R_base
        t = base_position_interp[0] - R @ gt_position_interp[0]

        # Apply transformation to base position and orientation
        for i in range(len(common_timestamps)):
            gt_position_interp[i] = R @ gt_position_interp[i] + t
            gt_orientation_interp[i] = rotation_matrix_to_quaternion(R @ quaternion_to_rotation_matrix(gt_orientation_interp[i]))

        # Print transformation details
        print("Rotation matrix from gt to base:")
        print(R)
        print("\nTranslation vector from gt to base:")
        print(t)
    else:
        print("Not spatially aligning data")

    return common_timestamps, base_position_interp, base_orientation_interp, gt_position_interp, gt_orientation_interp
def filter(imu_measurements, joint_measurements, force_torque_measurements, base_pose_ground_truth, serow_framework, state, align = False):
    base_positions = []
    base_orientations = []
    base_timestamps = []
    cumulative_rewards = {}
    for cf in state.get_contacts_frame():
        cumulative_rewards[cf] = []

    for imu, joint, ft, gt in zip(imu_measurements, joint_measurements, force_torque_measurements, base_pose_ground_truth):
        action = {}
        for cf in state.get_contacts_frame():
            action[cf] = np.ones(2)

        timestamp, state, rewards, _ = run_step(imu, joint, ft, gt, serow_framework, state, action)
        base_timestamps.append(timestamp)
        base_positions.append(state.get_base_position())
        base_orientations.append(state.get_base_orientation()) 
        for cf in rewards:
            if rewards[cf] is not None:
                cumulative_rewards[cf].append(rewards[cf])

    # Convert to numpy arrays
    base_position = np.array(base_positions)
    base_orientation = np.array(base_orientations)
    base_timestamps = np.array(base_timestamps)
    cumulative_rewards = {cf: np.array(cumulative_rewards[cf]) for cf in state.get_contacts_frame()}
    # Print evaluation metrics
    print("\nPolicy Evaluation Metrics:")
    for cf in state.get_contacts_frame():
        print(f"Average Cumulative Reward for {cf}: {np.mean(cumulative_rewards[cf]):.4f}")
        print(f"Max Cumulative Reward for {cf}: {np.max(cumulative_rewards[cf]):.4f}")
        print(f"Min Cumulative Reward for {cf}: {np.min(cumulative_rewards[cf]):.4f}")
        print("-------------------------------------------------")
    
    # Extract ground truth data with timestamps
    gt_timestamps = np.array([gt.timestamp for gt in base_pose_ground_truth])
    base_ground_truth_position = np.array([gt.position for gt in base_pose_ground_truth])
    base_ground_truth_orientation = np.array([gt.orientation for gt in base_pose_ground_truth])

    timestamps, base_position_aligned, base_orientation_aligned, gt_position_aligned, gt_orientation_aligned = sync_and_align_data(base_timestamps, base_position, base_orientation, gt_timestamps, base_ground_truth_position, base_ground_truth_orientation, align)

    return timestamps, base_position_aligned, base_orientation_aligned, gt_position_aligned, gt_orientation_aligned, cumulative_rewards

def plot_trajectories(timestamps, base_position, base_orientation, gt_position, gt_orientation, cumulative_rewards = None):
     # Plot the synchronized and aligned data
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, gt_position[:, 0], label="gt x")
    plt.plot(timestamps, base_position[:, 0], label="base x (aligned)")
    plt.plot(timestamps, gt_position[:, 1], label="gt y")
    plt.plot(timestamps, base_position[:, 1], label="base y (aligned)")
    plt.plot(timestamps, gt_position[:, 2], label="gt z")
    plt.plot(timestamps, base_position[:, 2], label="base z (aligned)")
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Base Position vs Ground Truth (Spatially Aligned)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, gt_orientation[:, 0], label="gt w")
    plt.plot(timestamps, base_orientation[:, 0], label="base w")
    plt.plot(timestamps, gt_orientation[:, 1], label="gt x")
    plt.plot(timestamps, base_orientation[:, 1], label="base x")
    plt.plot(timestamps, gt_orientation[:, 2], label="gt y")
    plt.plot(timestamps, base_orientation[:, 2], label="base y")
    plt.plot(timestamps, gt_orientation[:, 3], label="gt z")
    plt.plot(timestamps, base_orientation[:, 3], label="base z")
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Components')
    plt.title('Base Orientation vs Ground Truth')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot 3D trajectories
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_position[:, 0], gt_position[:, 1], gt_position[:, 2], 
            label='Ground Truth', color='blue')
    ax.plot(base_position[:, 0], base_position[:, 1], base_position[:, 2], 
            label='Base Position', color='green')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectories')
    ax.legend()
    plt.show()
    
    if cumulative_rewards is not None:
        n_cf = len(cumulative_rewards)
        fig, axes = plt.subplots(n_cf, 1, figsize=(12, 4*n_cf))
        if n_cf == 1:
            axes = [axes]  # Make axes iterable for single subplot case
        
        for ax, cf in zip(axes, cumulative_rewards):
            ax.plot(cumulative_rewards[cf])
            ax.set_xlabel('steps')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Cumulative Reward for {cf} over steps')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def plot_joint_states(joint_states):
    """Plot joint positions and velocities over time."""
    # Debug print to check joint states
    print(f"Number of joint states: {len(joint_states)}")
    if len(joint_states) == 0:
        print("Error: No joint states available")
        return
        
    # Get all unique joint names
    joint_names = list(joint_states[0].joints_position.keys())
    print(f"Joint names: {joint_names}")
    n_joints = len(joint_names)
    
    if n_joints == 0:
        print("Error: No joints found in the first joint state")
        return
        
    print(f"Number of joints: {n_joints}")
    
    # Create figure with subplots for positions
    fig_pos, axes_pos = plt.subplots(n_joints, 1, figsize=(12, 4*n_joints))
    fig_pos.suptitle('Joint Positions Over Time')
    
    # Create figure with subplots for velocities
    fig_vel, axes_vel = plt.subplots(n_joints, 1, figsize=(12, 4*n_joints))
    fig_vel.suptitle('Joint Velocities Over Time')
    
    # Create time array
    times = np.arange(len(joint_states))
    
    # Plot each joint's position and velocity
    for i, joint_name in enumerate(joint_names):
        # Extract position and velocity data for this joint
        positions = [state.joints_position[joint_name] for state in joint_states]
        velocities = [state.joints_velocity[joint_name] for state in joint_states]
        
        # Plot position
        axes_pos[i].plot(times, positions)
        axes_pos[i].set_ylabel('Position (rad)')
        axes_pos[i].set_title(f'{joint_name} Position')
        axes_pos[i].grid(True)
        
        # Plot velocity
        axes_vel[i].plot(times, velocities)
        axes_vel[i].set_ylabel('Velocity (rad/s)')
        axes_vel[i].set_title(f'{joint_name} Velocity')
        axes_vel[i].grid(True)
    
    # Add x-label to bottom subplot only
    axes_pos[-1].set_xlabel('Time Steps')
    axes_vel[-1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()

def plot_contact_states(contact_states):
    """Plot contact states over time.
    
    Args:
        contact_states: List of ContactState objects containing contact information
    """
    if not contact_states:
        print("No contact states to plot")
        return
        
    # Get all unique contact names from the first state
    contact_names = list(contact_states[0].contacts_status.keys())
    n_contacts = len(contact_names)
    
    if n_contacts == 0:
        print("No contacts found in the contact states")
        return
        
    # Create figure with subplots for status and probability
    fig, axes = plt.subplots(n_contacts, 2, figsize=(15, 4*n_contacts))
    fig.suptitle('Contact States Over Time')
    
    # Create time array
    times = np.arange(len(contact_states))
    
    # Plot each contact's status and probability
    for i, contact_name in enumerate(contact_names):
        # Extract status and probability data for this contact
        statuses = [state.contacts_status[contact_name] for state in contact_states]
        probabilities = [state.contacts_probability[contact_name] for state in contact_states]
        
        # Plot status
        ax_status = axes[i, 0] if n_contacts > 1 else axes[0]
        ax_status.plot(times, statuses, 'b-', label='Status')
        ax_status.set_ylabel('Contact Status')
        ax_status.set_title(f'{contact_name} Status')
        ax_status.set_ylim(-0.1, 1.1)  # Binary values
        ax_status.grid(True)
        
        # Plot probability
        ax_prob = axes[i, 1] if n_contacts > 1 else axes[1]
        ax_prob.plot(times, probabilities, 'r-', label='Probability')
        ax_prob.set_ylabel('Contact Probability')
        ax_prob.set_title(f'{contact_name} Probability')
        ax_prob.set_ylim(-0.1, 1.1)  # Probability range
        ax_prob.grid(True)
    
    # Add x-label to bottom subplots only
    if n_contacts > 1:
        axes[-1, 0].set_xlabel('Time Steps')
        axes[-1, 1].set_xlabel('Time Steps')
    else:
        axes[0].set_xlabel('Time Steps')
        axes[1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()

def plot_contact_forces_and_torques(contact_states):
    """Plot contact forces and torques over time.
    
    Args:
        contact_states: List of ContactState objects containing contact information
    """
    if not contact_states:
        print("No contact states to plot")
        return
        
    # Get all unique contact names from the first state
    contact_names = list(contact_states[0].contacts_force.keys())
    n_contacts = len(contact_names)
    
    if n_contacts == 0:
        print("No contacts found in the contact states")
        return
        
    # Create figure with subplots for forces and torques
    fig, axes = plt.subplots(n_contacts, 2, figsize=(15, 4*n_contacts))
    fig.suptitle('Contact Forces and Torques Over Time')
    
    # Create time array
    times = np.arange(len(contact_states))
    
    # Plot each contact's forces and torques
    for i, contact_name in enumerate(contact_names):
        # Extract force and torque data for this contact
        forces_x = [state.contacts_force[contact_name][0] for state in contact_states]
        forces_y = [state.contacts_force[contact_name][1] for state in contact_states]
        forces_z = [state.contacts_force[contact_name][2] for state in contact_states]
        
        # Plot forces
        ax_force = axes[i, 0] if n_contacts > 1 else axes[0]
        ax_force.plot(times, forces_x, 'r-', label='Fx')
        ax_force.plot(times, forces_y, 'g-', label='Fy')
        ax_force.plot(times, forces_z, 'b-', label='Fz')
        ax_force.set_ylabel('Force (N)')
        ax_force.set_title(f'{contact_name} Forces')
        ax_force.grid(True)
        ax_force.legend()
        
        # Plot torques if available
        ax_torque = axes[i, 1] if n_contacts > 1 else axes[1]
        if hasattr(contact_states[0], 'contacts_torque') and contact_states[0].contacts_torque:
            torques_x = [state.contacts_torque[contact_name][0] for state in contact_states]
            torques_y = [state.contacts_torque[contact_name][1] for state in contact_states]
            torques_z = [state.contacts_torque[contact_name][2] for state in contact_states]
            
            ax_torque.plot(times, torques_x, 'r-', label='Tx')
            ax_torque.plot(times, torques_y, 'g-', label='Ty')
            ax_torque.plot(times, torques_z, 'b-', label='Tz')
            ax_torque.set_ylabel('Torque (Nm)')
            ax_torque.set_title(f'{contact_name} Torques')
            ax_torque.grid(True)
            ax_torque.legend()
        else:
            ax_torque.text(0.5, 0.5, 'No torque data available', 
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=ax_torque.transAxes)
            ax_torque.set_title(f'{contact_name} Torques')
    
    # Add x-label to bottom subplots only
    if n_contacts > 1:
        axes[-1, 0].set_xlabel('Time Steps')
        axes[-1, 1].set_xlabel('Time Steps')
    else:
        axes[0].set_xlabel('Time Steps')
        axes[1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Read the measurement mcap file
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")
   
    # Plot joint states
    # plot_joint_states(joint_states)

    # Plot contact states
    # plot_contact_states(contact_states)
    # plot_contact_forces_and_torques(contact_states)

    # offset = len(imu_measurements) - len(base_states) 
    # print(f"sample offset: {offset}")
    # imu_measurements = imu_measurements[offset:]
    # joint_measurements = joint_measurements[offset:]
    # force_torque_measurements = force_torque_measurements[offset:]
    # base_pose_ground_truth = base_pose_ground_truth[offset:]

    # initialize at a different time
    # new_offset = 11000
    # base_states = base_states[new_offset:]
    # contact_states = contact_states[new_offset:]
    # joint_states = joint_states[new_offset:]
    # imu_measurements = imu_measurements[new_offset:]
    # joint_measurements = joint_measurements[new_offset:]
    # force_torque_measurements = force_torque_measurements[new_offset:]
    # base_pose_ground_truth = base_pose_ground_truth[new_offset:]

    # Initialize SEROW
    serow_framework = serow.Serow()
    serow_framework.initialize("go2_rl.json")
    state = serow_framework.get_state(allow_invalid=True)
    state.set_joint_state(joint_states[0])
    state.set_base_state(base_states[0])  
    state.set_contact_state(contact_states[0])
    serow_framework.set_state(state)

    # Run SEROW
    timestamps, base_position, base_orientation, gt_position, gt_orientation, cumulative_rewards = filter(imu_measurements, joint_measurements, force_torque_measurements, base_pose_ground_truth, serow_framework, state, align=False)
    
    # Plot the trajectories
    plot_trajectories(timestamps, base_position, base_orientation, gt_position, gt_orientation, cumulative_rewards)
   

