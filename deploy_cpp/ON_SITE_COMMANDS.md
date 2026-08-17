# First-day command card

Commands below are for an Ubuntu machine with the complete deployment package.
They do not enable user control or publish `LowCmd`.

```bash
# 1) Inspect network interfaces with the technical adviser.
ip link

# 2) Build the read-only tools.
cd g1_race_deployment_package/deploy_cpp
cmake -S . -B build -DUNITREE_SDK2_DIR=../third_party/unitree_sdk2
cmake --build build --target g1_read_only g1_joint_map_report -j2

# 3) Print the expected 14-joint mapping; save this result.
./build/g1_joint_map_report | tee joint_map_report.txt

# 4) With the adviser-confirmed interface only, receive low state.
# This executable has no command publisher.
./build/g1_read_only <interface-name> | tee read_only.log
```

Stop with `Ctrl-C`. If no messages arrive, stop and verify interface, robot IP,
DDS domain and official SDK environment with the technical adviser. Do not
switch user control or run the official motion examples to troubleshoot.
