#!/bin/bash

# Add imports to KinematicMeasurement.py
echo -e "from foxglove.Matrix3 import Matrix3\n
from foxglove.Quaternion import Quaternion\n
from foxglove.Time import Time\n
from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/KinematicMeasurement.py)" > ../build/generated/foxglove/KinematicMeasurement.py

# Add imports to ImuMeasurement.py
echo -e "from foxglove.Matrix3 import Matrix3\n
from foxglove.Quaternion import Quaternion\n
from foxglove.Time import Time\n
from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/ImuMeasurement.py)" > ../build/generated/foxglove/ImuMeasurement.py

# Add imports to BaseState.py
echo -e "from foxglove.Matrix3 import Matrix3\n
from foxglove.Quaternion import Quaternion\n
from foxglove.Time import Time\n
from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/BaseState.py)" > ../build/generated/foxglove/BaseState.py

# Add imports to FrameTransform.py
echo -e "from foxglove.Quaternion import Quaternion\n
from foxglove.Time import Time\n
from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/FrameTransform.py)" > ../build/generated/foxglove/FrameTransform.py 

# Add imports to ForceTorqueMeasurements.py
echo -e "from foxglove.Time import Time\n
from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/ForceTorqueMeasurements.py)" > ../build/generated/foxglove/ForceTorqueMeasurements.py
 
# Add imports to JointMeasurements.py
echo -e "from foxglove.Time import Time\n$(cat ../build/generated/foxglove/JointMeasurements.py)" > ../build/generated/foxglove/JointMeasurements.py 

# Add imports to ContactState.py
echo -e "from foxglove.Time import Time\n
from foxglove.Contact import Contact\n$(cat ../build/generated/foxglove/ContactState.py)" > ../build/generated/foxglove/ContactState.py

# Add imports to Contact.py
echo -e "from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/Contact.py)" > ../build/generated/foxglove/Contact.py

# Add imports to JointState.py
echo -e "from foxglove.Time import Time\n$(cat ../build/generated/foxglove/JointState.py)" > ../build/generated/foxglove/JointState.py 
