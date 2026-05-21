# FSDP: Fast and Safe Data-Driven Overtaking Trajectory Planning

<p align="center">
  <a href="https://arxiv.org/abs/2503.06075">
    <img src="https://img.shields.io/badge/arXiv.org-2503.06075-b31b1b" alt="arXiv e-print badge">
  </a>
  <img alt="ROS Noetic" src="https://img.shields.io/badge/ROS-Noetic-22314E">
  <img alt="Python 3.8" src="https://img.shields.io/badge/Python-3.8-3776AB">
  <img alt="Sparse GP Prediction" src="https://img.shields.io/badge/Sparse%20GP-Prediction-0F766E">
  <img alt="Bi-level QP" src="https://img.shields.io/badge/Bi--level%20QP-Control-F59E0B">
</p>

`fsdp` is the DDRX implementation of
[FSDP: Fast and Safe Data-Driven Overtaking Trajectory Planning for Head-to-Head Autonomous Racing Competitions](https://arxiv.org/abs/2503.06075).
It uses sparse Gaussian Process opponent prediction, collision-risk checking, and a bi-level QP planning pipeline: polynomial fitting produces a rough overtaking trajectory, then a Frenet-frame MPC QP refines it with kinematic and safety constraints.

<img src="./misc/fsdp_framework.png" alt="FSDP framework" style="width: 100%;"/>

## Demo

<table>
  <tr>
    <td align="center"><strong>Simulation / Planning View</strong></td>
    <td align="center"><strong>Physical F1TENTH Platform</strong></td>
  </tr>
  <tr>
    <td><img src="./media/gifs/fsdp_simulation_overtake.gif" alt="FSDP simulation overtaking demo" width="100%"/></td>
    <td><img src="./media/gifs/fsdp_physical_overtake.gif" alt="FSDP physical overtaking demo" width="100%"/></td>
  </tr>
</table>

## Package Layout

- `launch/fsdp.launch`: package-level launch file, mirroring the `ddrx_spliner_multi` workflow.
- `src/sqp/`: main FSDP overtaking planner nodes; `sqp` is the legacy folder and node naming.
- `src/soc/`: collision prediction and dynamic collision tuning.
- `src/gp/`: opponent trajectory projection and GP prediction nodes.
- `src/waypoint/`: `/global_waypoints_updated` publisher.
- `src/common/`: shared Frenet, polynomial fitting, and QP helpers.
- `src/mpc/`: vehicle model, MPC controllers, and pybind11 C++ extension.
- `cfg/*.cfg`: dynamic reconfigure parameters for collision prediction and SQP tuning.

## Installation

This repository contains only the FSDP planner package. Install the base race stack first by following
[ForzaETH/race_stack](https://github.com/ForzaETH/race_stack), then clone this repository into the `planner/` folder:

```bash
cd <race_stack_repo>/planner
git clone git@github.com:ZJU-DDRX/FSDP.git fsdp
```

## Build

```bash
cd <race_stack_ws>
catkin build fsdp
```

The package now builds the local `mpc_builder` extension automatically during `catkin build` and installs the runtime Python packages into the catkin devel Python path. A quick verification is:

```bash
python3 - <<'PY'
import mpc_builder
from common.qp_fit import QPFit
from mpc.mpc_tracking_controller_ca import MPC_Tracking_Controller
print("fsdp imports OK:", mpc_builder.__file__)
PY
```

If you clone this stack again, keep submodules enabled. The GP node depends on the local `ccma` package:

```bash
git submodule update --init --recursive
pip install <race_stack_repo>/f110_utils/libs/ccma
```

## Quick Start

Start the base simulator:

```bash
cd <race_stack_ws>
roslaunch stack_master base_system.launch sim:=True racecar_version:=SIM map_name:=f rviz:=false
```

In a second terminal, launch head-to-head with the FSDP planner:

```bash
roslaunch stack_master headtohead.launch perception:=False planner:=fsdp
```

In a third terminal, publish a simulated opponent obstacle:

```bash
roslaunch obstacle_publisher obstacle_publisher.launch speed_scaler:=0.3
```

Open `rqt_reconfigure` and enable the overtaking sectors if the state machine is still staying on the raceline:

```bash
rqt_reconfigure
```

## What To Watch

These topics are the fastest way to confirm the pipeline is alive:

```bash
rostopic hz /perception/obstacles
rostopic hz /proj_opponent_trajectory
rostopic hz /opponent_trajectory
rostopic hz /collision_prediction/obstacles
rostopic hz /global_waypoints_updated
rostopic hz /planner/avoidance/otwpnts
```

RViz markers:

- `/opponent_traj_markerarray`
- `/planner/avoidance/markers_sqp`
- `/planner/avoidance/df_markers`
- `/planner/avoidance/mpc_markers`
- `/planner/avoidance/raw_traj_markers`

## Runtime Notes

- `launch_gp:=true` starts the full opponent prediction chain and requires `ccma`, `torch`, `gpytorch`, `scikit-learn`, and `pandas`.
- The planner waits for `/global_waypoints`, `/global_waypoints_updated`, and obstacle topics, so a direct package launch without the base stack may appear idle.
- Generated build products under `build/`, `src/mpc_builder*.so`, debug data, and spreadsheet outputs are ignored by `.gitignore`.
- `stack_master/launch/headtohead.launch` includes this package launch when `planner:=fsdp`.

## Citation

```bibtex
@inproceedings{hu2025fsdp,
  title={Fsdp: Fast and safe data-driven overtaking trajectory planning for head-to-head autonomous racing competitions},
  author={Hu, Cheng and Huang, Jihao and Mao, Wule and Fu, Yonghao and Chi, Xuemin and Qin, Haotong and Baumann, Nicolas and Liu, Zhitao and Magno, Michele and Xie, Lei},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5824--5831},
  year={2025},
  organization={IEEE}
}
```
