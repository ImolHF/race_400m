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
    {0, 0, "left_hip_pitch", 0.25F}, {1, 6, "right_hip_pitch", 0.25F},
    {2, 1, "left_hip_roll", 0.25F}, {3, 7, "right_hip_roll", 0.25F},
    {4, 2, "left_hip_yaw", 0.25F}, {5, 8, "right_hip_yaw", 0.25F},
    {6, 3, "left_knee", 0.25F}, {7, 9, "right_knee", 0.25F},
    {8, 15, "left_shoulder_pitch", 0.18F}, {9, 22, "right_shoulder_pitch", 0.18F},
    {10, 4, "left_ankle_pitch", 0.20F}, {11, 10, "right_ankle_pitch", 0.20F},
    {12, 5, "left_ankle_roll", 0.20F}, {13, 11, "right_ankle_roll", 0.20F},
}};
