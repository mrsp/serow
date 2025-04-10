#!/bin/bash

# Add imports to KinematicMeasurement.py
echo -e "from foxglove.Matrix3 import Matrix3\n
from foxglove.Quaternion import Quaternion\n
from foxglove.Vector3 import Vector3\n
from foxglove.StringBoolEntry import StringBoolEntry\n
from foxglove.StringDoubleEntry import StringDoubleEntry\n
from foxglove.StringVector3Entry import StringVector3Entry\n
from foxglove.StringMatrix3Entry import StringMatrix3Entry\n
from foxglove.StringQuaternionEntry import StringQuaternionEntry\n$(cat ../build/generated/foxglove/KinematicMeasurement.py)" > ../build/generated/foxglove/KinematicMeasurement.py

# Add imports to StringVector3Entry.py
echo -e "from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/StringVector3Entry.py)" > ../build/generated/foxglove/StringVector3Entry.py

# Add imports to StringMatrix3Entry.py
echo -e "from foxglove.Matrix3 import Matrix3\n$(cat ../build/generated/foxglove/StringMatrix3Entry.py)" > ../build/generated/foxglove/StringMatrix3Entry.py

# Add imports to ImuMeasurement.py
echo -e "from foxglove.Matrix3 import Matrix3\n
from foxglove.Quaternion import Quaternion\n
from foxglove.Time import Time\n
from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/ImuMeasurement.py)" > ../build/generated/foxglove/ImuMeasurement.py

# Add imports to BaseState.py
echo -e "from foxglove.Matrix3 import Matrix3\n
from foxglove.Quaternion import Quaternion\n
from foxglove.Time import Time\n
from foxglove.Vector3 import Vector3\n
from foxglove.ContactPosition import ContactPosition\n
from foxglove.ContactOrientation import ContactOrientation\n
from foxglove.ContactPositionCov import ContactPositionCov\n
from foxglove.ContactOrientationCov import ContactOrientationCov\n$(cat ../build/generated/foxglove/BaseState.py)" > ../build/generated/foxglove/BaseState.py

# Add imports to ContactPosition.py
echo -e "from foxglove.Vector3 import Vector3\n$(cat ../build/generated/foxglove/ContactPosition.py)" > ../build/generated/foxglove/ContactPosition.py

# Add imports to ContactOrientation.py
echo -e "from foxglove.Quaternion import Quaternion\n$(cat ../build/generated/foxglove/ContactOrientation.py)" > ../build/generated/foxglove/ContactOrientation.py

# Add imports to ContactPositionCov.py
echo -e "from foxglove.Matrix3 import Matrix3\n$(cat ../build/generated/foxglove/ContactPositionCov.py)" > ../build/generated/foxglove/ContactPositionCov.py

# Add imports to ContactOrientationCov.py
echo -e "from foxglove.Matrix3 import Matrix3\n$(cat ../build/generated/foxglove/ContactOrientationCov.py)" > ../build/generated/foxglove/ContactOrientationCov.py 
