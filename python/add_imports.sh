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
