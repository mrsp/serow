# Install mcap
Use this repo to install mcap for cpp. 

https://github.com/olympus-robotics/mcap_builder 


# Install ZSTD compression from source
```
$ git clone https://github.com/facebook/zstd.git
$ cd zstd
$ cd build/cmake
$ cmake .
$ sudo make install
```



Compile and run the go2_mujoco_test

```
$ cd serow/evaluation/mujoco_test/
$ mkdir build 
$ cd build && cmake ..
$ make -j8
$ ./go2_mujoco_test
```
This saves serow predictions to the mujoco_data/dataset where dataset is specified in the test_config.json
