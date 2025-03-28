## Trajectory Visualization Overview
When running a script at test/build for a robot dataset, the serow base state estimates (position + orientation) are saved at *results/<name>.txt* if the flag kStorePredictions is true. In order to visualize the predictions follow the instructions below

## Installation
In order to visualize the estimated trajectory you will the evo package:
```
pip install evo --upgrade --no-binary evo
```
## Visualization
Navigate to the serow/test/results/ folder and run:
```
evo_traj tum <result_filename>.txt  <result_filename>.txt --plot --plot_mode=xyz
```
This uses the estimations as reference if the ground truth is not available. If it is, swap the second txt with the ground truth poses. 

Example for visualizing go2 serow estimates:
```
evo_traj tum go2_serow_estimates.txt go2_serow_estimates.txt --plot --plot_mode=xyz
```
