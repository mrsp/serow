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

bash add_imports.sh
