// SDK-only, read-only G1 LowState snapshot writer.
// It has no ROS dependency, no publisher, no LowCmd, and no control client.
// Output CSV layout: unix_time_s, q[29], dq[29], imu_wxyz[4], gyro[3], mode
// (67 comma-separated values in total).

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
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

struct Snapshot {
  std::array<float, 29> q{};
  std::array<float, 29> dq{};
  std::array<float, 4> quaternion_wxyz{};
  std::array<float, 3> gyro{};
  float mode_machine{0.0F};
  double unix_time_s{0.0};
  bool received{false};
};

std::mutex g_mutex;
Snapshot g_snapshot;

double UnixTimeSeconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

void OnLowState(const void* raw) {
  const auto& state = *static_cast<const LowState_*>(raw);
  std::lock_guard<std::mutex> lock(g_mutex);
  for (int i = 0; i < 29; ++i) {
    g_snapshot.q.at(i) = state.motor_state().at(i).q();
    g_snapshot.dq.at(i) = state.motor_state().at(i).dq();
  }
  const auto& imu = state.imu_state();
  for (int i = 0; i < 4; ++i) g_snapshot.quaternion_wxyz.at(i) = imu.quaternion().at(i);
  for (int i = 0; i < 3; ++i) g_snapshot.gyro.at(i) = imu.gyroscope().at(i);
  g_snapshot.mode_machine = static_cast<float>(state.mode_machine());
  g_snapshot.unix_time_s = UnixTimeSeconds();
  g_snapshot.received = true;
}

bool WriteSnapshotAtomically(const std::string& path, const Snapshot& snapshot) {
  const std::string temporary = path + ".tmp";
  std::ofstream file(temporary, std::ios::out | std::ios::trunc);
  if (!file) return false;
  file << std::fixed << std::setprecision(9) << snapshot.unix_time_s;
  const auto write = [&file](const auto& values) {
    for (const float value : values) file << ',' << value;
  };
  write(snapshot.q);
  write(snapshot.dq);
  write(snapshot.quaternion_wxyz);
  write(snapshot.gyro);
  file << ',' << snapshot.mode_machine << '\n';
  file.close();
  return std::rename(temporary.c_str(), path.c_str()) == 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: g1_lowstate_snapshot <network-interface> [output-csv]\n";
    return 2;
  }
  const std::string output = argc == 3 ? argv[2] : "/tmp/g1_shadow_lowstate.csv";
  ChannelFactory::Instance()->Init(0, argv[1]);
  ChannelSubscriberPtr<LowState_> subscriber(new ChannelSubscriber<LowState_>(kLowStateTopic));
  subscriber->InitChannel(OnLowState, 1);
  std::cout << "READ-ONLY: writing SDK LowState snapshots to " << output
            << ". No ROS publisher, LowCmd, or robot-control client is constructed. Ctrl-C exits.\n";

  std::uint64_t writes = 0;
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 50 Hz snapshots
    Snapshot copy;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      copy = g_snapshot;
    }
    if (!copy.received) continue;
    if (!WriteSnapshotAtomically(output, copy)) {
      std::cerr << "Failed to write " << output << '\n';
      continue;
    }
    if (++writes % 250 == 1) {
      std::cout << "snapshots=" << writes << " mode_machine=" << copy.mode_machine << '\n';
    }
  }
}
