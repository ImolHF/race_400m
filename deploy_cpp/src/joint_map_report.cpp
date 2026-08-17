#include <iostream>
#include "joint_map.hpp"

int main() {
  std::cout << "G1 race policy joint-map report (NO ROBOT CONNECTION)\n";
  std::cout << "action,motor_id,sdk_joint_name,action_scale_rad,physical_direction_confirmed\n";
  for (const auto& joint : kPolicyJoints) {
    std::cout << joint.action_index << ',' << joint.motor_id << ',' << joint.name << ','
              << joint.action_scale << ",NO\n";
  }
  std::cout << "mode_machine: copy the live LowState value (reported 5), never hard-code it.\n";
}
