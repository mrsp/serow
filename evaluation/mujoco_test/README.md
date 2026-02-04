# Install mcap
https://github.com/olympus-robotics/mcap_builder 

Use this repo to install mcap for cpp. Compile and run the go2_mujoco_test

```
$ mkdir build 
$ cd build && cmake ..
$ make -j8
$ ./go2_mujoco_test
```
This saves serow predictions to the mujoco_data/dataset where dataset is specified in the test_config.json
