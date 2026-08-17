#pragma once

#include <array>
#include <cstddef>

struct PolicyJoint {
  int action_index;
  int motor_id;
  const char* name;
  float action_scale;
};

// Verified against the official SDK2 G1 enum. Physical direction still must
// be confirmed under suspension with the on-site technical adviser.
constexpr std::array<PolicyJoint, 14> kPolicyJoints{{
    {0, 0, "left_hip_pitch", 0.25F}, {1, 1, "left_hip_roll", 0.25F},
    {2, 2, "left_hip_yaw", 0.25F}, {3, 3, "left_knee", 0.25F},
    {4, 4, "left_ankle_pitch", 0.20F}, {5, 5, "left_ankle_roll", 0.20F},
    {6, 6, "right_hip_pitch", 0.25F}, {7, 7, "right_hip_roll", 0.25F},
    {8, 8, "right_hip_yaw", 0.25F}, {9, 9, "right_knee", 0.25F},
    {10, 10, "right_ankle_pitch", 0.20F}, {11, 11, "right_ankle_roll", 0.20F},
    {12, 15, "left_shoulder_pitch", 0.18F}, {13, 22, "right_shoulder_pitch", 0.18F},
}};
