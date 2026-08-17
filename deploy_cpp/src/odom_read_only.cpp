// Read-only ROS 2 odometry probe for the G1 on-board computer.
// It has no publisher, service client, Unitree command object, or policy inference.

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace {
class OdomReadOnly final : public rclcpp::Node {
 public:
  explicit OdomReadOnly(const std::string& topic) : Node("g1_odom_read_only") {
    using std::placeholders::_1;
    subscription_ = create_subscription<nav_msgs::msg::Odometry>(
        topic, rclcpp::QoS(10).reliable(), std::bind(&OdomReadOnly::OnOdom, this, _1));
    RCLCPP_INFO(get_logger(), "READ-ONLY: subscribing to %s; no publisher is constructed.", topic.c_str());
  }

 private:
  void OnOdom(const nav_msgs::msg::Odometry::SharedPtr message) {
    ++messages_;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_print_ < std::chrono::milliseconds(200)) return;
    last_print_ = now;
    const auto& p = message->pose.pose.position;
    const auto& q = message->pose.pose.orientation;  // ROS order: x,y,z,w.
    const auto& v = message->twist.twist.linear;
    const auto& w = message->twist.twist.angular;
    RCLCPP_INFO(get_logger(),
      "messages=%zu frame=%s child=%s pos_odom_xy=(%.3f, %.3f) z=%.3f "
      "quat_ros_xyzw=(%.5f, %.5f, %.5f, %.5f) linear=(%.3f, %.3f, %.3f) angular=(%.3f, %.3f, %.3f)",
      messages_, message->header.frame_id.c_str(), message->child_frame_id.c_str(), p.x, p.y, p.z,
      q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z);
  }
  std::size_t messages_{0};
  std::chrono::steady_clock::time_point last_print_{std::chrono::steady_clock::now()};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
};
}  // namespace

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  const std::string topic = argc >= 2 ? argv[1] : "/state_estimator/odom_pelvis";
  rclcpp::spin(std::make_shared<OdomReadOnly>(topic));
  rclcpp::shutdown();
  return 0;
}
