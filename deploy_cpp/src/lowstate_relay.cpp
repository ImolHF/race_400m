// Read-only bridge from the SDK2 G1 LowState DDS stream to a *diagnostic*
// ROS 2 topic. It never constructs LowCmd, a robot client, or a control switch.
//
// Topic: /g1_shadow/lowstate_f32
// Float32MultiArray layout (66 values):
//   [0:29) q, [29:58) dq, [58:62) imu quaternion (SDK w,x,y,z),
//   [62:65) gyro (rad/s), [65] mode_machine.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using unitree_hg::msg::dds_::LowState_;

namespace {
constexpr char kLowStateTopic[] = "rt/lowstate";
constexpr char kDiagnosticTopic[] = "/g1_shadow/lowstate_f32";
rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr g_publisher;
std::atomic<std::uint64_t> g_messages{0};

void OnLowState(const void* raw) {
  const auto& state = *static_cast<const LowState_*>(raw);
  std_msgs::msg::Float32MultiArray out;
  out.data.resize(66);
  for (int i = 0; i < 29; ++i) {
    out.data.at(i) = state.motor_state().at(i).q();
    out.data.at(29 + i) = state.motor_state().at(i).dq();
  }
  const auto& imu = state.imu_state();
  for (int i = 0; i < 4; ++i) out.data.at(58 + i) = imu.quaternion().at(i);
  for (int i = 0; i < 3; ++i) out.data.at(62 + i) = imu.gyroscope().at(i);
  out.data.at(65) = static_cast<float>(state.mode_machine());
  g_publisher->publish(out);

  const auto n = ++g_messages;
  if (n % 500 == 1) {
    std::cout << "relayed=" << n << " samples to " << kDiagnosticTopic
              << " (66 floats; diagnostic only; no LowCmd exists)\n";
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: g1_lowstate_relay <network-interface>, e.g. g1_lowstate_relay eth0\n";
    return 2;
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("g1_lowstate_read_only_relay");
  g_publisher = node->create_publisher<std_msgs::msg::Float32MultiArray>(kDiagnosticTopic, 10);

  ChannelFactory::Instance()->Init(0, argv[1]);
  ChannelSubscriberPtr<LowState_> subscriber(new ChannelSubscriber<LowState_>(kLowStateTopic));
  subscriber->InitChannel(OnLowState, 1);
  std::cout << "READ-ONLY: subscribed to " << kLowStateTopic << "; publishing diagnostic state to "
            << kDiagnosticTopic << ". No LowCmd publisher or robot-control client is constructed. Ctrl-C exits.\n";
  while (rclcpp::ok()) std::this_thread::sleep_for(std::chrono::seconds(1));
  rclcpp::shutdown();
  return 0;
}
