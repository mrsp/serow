# Python Bindings

## Install the python package

Open a terminal in the current directory

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.bashrc
```

Install the dependencies

```
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
```

## Tests
Run the example with: 

```
python3 serow/example.py
```

## Train with DDPG

KinematicMeasurement.py

from foxglove.Matrix3 import Matrix3
from foxglove.Quaternion import Quaternion
from foxglove.Vector3 import Vector3
from foxglove.StringBoolEntry import StringBoolEntry
from foxglove.StringDoubleEntry import StringDoubleEntry
from foxglove.StringVector3Entry import StringVector3Entry
from foxglove.StringMatrix3Entry import StringMatrix3Entry
from foxglove.StringQuaternionEntry import StringQuaternionEntry

StringVector3Entry.py
from foxglove.Vector3 import Vector3

StringMatrix3Entry.py
from foxglove.Matrix3 import Matrix3

ImuMeasurement.py
from foxglove.Matrix3 import Matrix3
from foxglove.Quaternion import Quaternion
from foxglove.Time import Time
from foxglove.Vector3 import Vector3

BaseState.py
from foxglove.Matrix3 import Matrix3
from foxglove.Quaternion import Quaternion
from foxglove.Time import Time
from foxglove.Vector3 import Vector3
from foxglove.ContactPosition import ContactPosition
from foxglove.ContactOrientation import ContactOrientation
from foxglove.ContactPositionCov import ContactPositionCov
from foxglove.ContactOrientationCov import ContactOrientationCov

ContactPosition.py
from foxglove.Vector3 import Vector3

ContactOrientation.py
from foxglove.Quaternion import Quaternion

ContactPositionCov.py
from foxglove.Matrix3 import Matrix3

ContactOrientationCov.py
from foxglove.Matrix3 import Matrix3

