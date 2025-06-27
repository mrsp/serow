# Python Bindings

## Install the Python Package

Open a terminal in the current directory:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
source ~/.bashrc
```

Install the dependencies:

```
pip3 install -r requirements.txt
```

# Build SEROW with the Python Schemas 

Remove existing build (if present):

```
cd SEROW_PATH
rm -rf build
mkdir build && cd build
```

Generate the Python schemas:

```
cmake .. -DGENERATE_PYTHON_SCHEMAS=ON
make -j8
sudo make install
```

Fix Python schemas and build the package:

```
cd SEROW_PATH/python
./add_imports.sh
python3 setup.py build_ext --inplace
```

## Tests

Run the example with: 

```
python3 serow/example.py
```

## Train with DDPG

Run the training loop:

```
python3 serow/train_ddpg.py
```

After training, the best and final (converged) policy will be stored in the `policy/ddpg` folder.


## Inference with DDPG


Re-running:

```
python3 serow/train_ddpg.py
```

will load the final policy and do inference. 

Alternatively, the user can employ ONNX for the inference step by running:

```
python3 serow/inference_ddpg.py
```

# Real-time Inference with SEROW

Build SEROW with ONNX support:

```
cd SEROW_PATH
rm -rf build
mkdir build && cd build
```

Enable ONNX support:

```
cmake .. -DGENERATE_PYTHON_SCHEMAS=ON -DONNX=ON
make -j8
sudo make install
```

## Run a test with ONNX support

Remove existing build (if present):

```
cd SEROW_PATH/evaluation/mujoco_test
rm -rf build
mkdir build && cd build
```

Enable ONNX support:

```
cmake ..  -DONNX=ON
make -j8
./go2_mujoco_test
```

Visualize the results with:

```
cd ..
python3 serow_viz.py
```
