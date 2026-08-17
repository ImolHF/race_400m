#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using unitree_hg::msg::dds_::LowState_;

namespace {
constexpr char kLowStateTopic[] = "rt/lowstate";
std::atomic<unsigned long> g_messages{0};

void OnLowState(const void* raw) {
  const auto& state = *static_cast<const LowState_*>(raw);
  const auto number = ++g_messages;
  if (number % 100 != 1) return;  // print about once per second on a 100 Hz stream

  std::cout << "messages=" << number << " mode_machine="
            << static_cast<unsigned>(state.mode_machine()) << " q[0..28]= ";
  std::cout << std::fixed << std::setprecision(3);
  for (int i = 0; i < 29; ++i) std::cout << state.motor_state().at(i).q() << (i == 28 ? "" : ",");
  const auto& imu = state.imu_state();
  std::cout << "\nimu_quaternion(raw SDK order)=";
  for (float value : imu.quaternion()) std::cout << value << ',';
  std::cout << " gyro=";
  for (float value : imu.gyroscope()) std::cout << value << ',';
  std::cout << "\nREAD-ONLY: no publisher is constructed and no motor command can be sent.\n";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: g1_read_only <network-interface>, e.g. g1_read_only eth0\n";
    return 2;
  }
  ChannelFactory::Instance()->Init(0, argv[1]);
  ChannelSubscriberPtr<LowState_> subscriber(new ChannelSubscriber<LowState_>(kLowStateTopic));
  subscriber->InitChannel(OnLowState, 1);
  std::cout << "Subscribed to " << kLowStateTopic << " on " << argv[1]
            << ". Ctrl-C exits. This program has no command publisher.\n";
  while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
}
