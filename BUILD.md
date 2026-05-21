# FSDP Build Notes

`fsdp` uses a local pybind11/C++ extension named `mpc_builder`.
The extension is now built automatically by `catkin build fsdp`; no manual pre-build step is needed.

## Dependencies

The original race-stack Docker image installs the package dependencies through:

```bash
.devcontainer/.install_utils/linux_req_car.txt
.devcontainer/.install_utils/requirements.txt
.devcontainer/.install_utils/post_installation.sh
```

For native installation, make sure these are available:

```bash
sudo apt-get install libeigen3-dev python3-numpy python3-pybind11
pip install -r /path/to/race-stack/.devcontainer/.install_utils/requirements.txt
pip install /path/to/race-stack/f110_utils/libs/ccma
```

## Build

```bash
cd ~/catkin_ws
catkin build fsdp
source devel/setup.bash
```

Verify the Python extension and helper modules:

```bash
python3 - <<'PY'
import mpc_builder
from common.qp_fit import QPFit
from mpc.mpc_tracking_controller_ca import MPC_Tracking_Controller
print("mpc_builder:", mpc_builder.__file__)
print("fsdp build OK")
PY
```

## Manual Extension Rebuild

Only use this if you are editing `src/mpc/cpp_mpc/mpc_builder.cpp` and want a quick local compile:

```bash
cd ~/catkin_ws/src/FSDP
python3 setup.py build_ext --inplace --force
```

After manual rebuild, run `catkin build fsdp` once so the `.so` is copied into the catkin devel Python path.
