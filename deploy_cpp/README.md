# G1 C++ on-site preparation

This folder deliberately contains **no executable policy-to-motor controller**.
It is the safe first-day scaffold built from the official `unitree_sdk2` G1
examples.

## Ubuntu build

Copy the entire deployment package to an Ubuntu computer with the official SDK
dependencies, then run:

```bash
cd g1_race_deployment_package/deploy_cpp
cmake -S . -B build -DUNITREE_SDK2_DIR=../third_party/unitree_sdk2
cmake --build build -j
./build/g1_joint_map_report
```

The official SDK2 repository documents Ubuntu 20.04, CMake, GCC, Make and its
listed dependency packages. Use the technical adviser's installed SDK/runtime
when it differs from the packaged source.

## On-site sequence

1. Run `g1_joint_map_report`; print or save the report.
2. Confirm the network interface with the adviser (for example `ip link`).
3. Run only `./build/g1_read_only <interface>`. It subscribes to `rt/lowstate`
   and constructs no command publisher.
4. Confirm all 29 q values change plausibly, `mode_machine=5` is received, and
   record the SDK's raw IMU quaternion ordering. Do **not** infer that order.
5. Only after the adviser confirms SDK control switching, joint directions,
   emergency stop, and suspension may a separately reviewed command program be
   created.

## Optional LibTorch check

After a compatible LibTorch installation is provided on the Ubuntu machine:

```bash
cmake -S . -B build -DUNITREE_SDK2_DIR=../third_party/unitree_sdk2 \
  -DENABLE_LIBTORCH_SHADOW=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
./build/g1_shadow_policy ../locked_elbow_main_84obs_14act_policy.pt
```

This loads the selected 84-observation/14-action TorchScript policy but does
not subscribe to, switch, or command the robot.
