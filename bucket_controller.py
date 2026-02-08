#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from custom_msgs.msg import Commands
import time

# -----------------------------
# State Definitions
# -----------------------------
class State:
    SEARCH = 0
    ALIGN = 1
    APPROACH = 2
    REACHED = 3


class BucketController(Node):
    def __init__(self):
        super().__init__("bucket_controller")

        # -----------------------------
        # Declare ROS Parameters
        # -----------------------------
        self.declare_parameter("kp_sway", 0.0)
        self.declare_parameter("kp_yaw", 0.0)

        self.declare_parameter("search_yaw_pwm", 1500)
        self.declare_parameter("approach_forward_pwm", 1500)

        self.declare_parameter("alignment_threshold", 0.0)
        self.declare_parameter("reach_distance", 0.0)

        self.declare_parameter("pwm_neutral", 1500)
        self.declare_parameter("pwm_range", 400)

        self.declare_parameter("visibility_timeout", 1.0)

        # -----------------------------
        # Load Parameters
        # -----------------------------
        self.kp_sway = self.get_parameter("kp_sway").value
        self.kp_yaw = self.get_parameter("kp_yaw").value

        self.search_yaw_pwm = self.get_parameter("search_yaw_pwm").value
        self.approach_forward_pwm = self.get_parameter("approach_forward_pwm").value

        self.alignment_threshold = self.get_parameter("alignment_threshold").value
        self.reach_distance = self.get_parameter("reach_distance").value

        self.pwm_neutral = self.get_parameter("pwm_neutral").value
        self.pwm_range = self.get_parameter("pwm_range").value

        self.visibility_timeout = self.get_parameter("visibility_timeout").value

        # -----------------------------
        # Internal State
        # -----------------------------
        self.state = State.SEARCH

        self.bucket_visible = False
        self.last_seen_time = 0.0

        # Perception errors (relative)
        self.bucket_x = 0.0   # horizontal error
        self.bucket_z = 0.0   # distance estimate

        # -----------------------------
        # ROS Interfaces
        # -----------------------------
        self.cmd_pub = self.create_publisher(
            Commands,
            "/master/commands",
            10
        )

        self.create_subscription(
            PoseStamped,
            "/perception/bucket",
            self.bucket_callback,
            10
        )

        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Bucket controller started (parameterized)")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def bucket_callback(self, msg):
        """
        Receives relative bucket information from perception.
        """
        self.bucket_visible = True
        self.last_seen_time = time.time()

        self.bucket_x = msg.pose.position.x
        self.bucket_z = msg.pose.position.z

    # -----------------------------
    # Main Control Loop
    # -----------------------------
    def control_loop(self):
        cmd = Commands()

        # Safety defaults
        cmd.mode = "ALT_HOLD"
        cmd.arm = 1

        cmd.forward = self.pwm_neutral
        cmd.lateral = self.pwm_neutral
        cmd.yaw = self.pwm_neutral
        cmd.thrust = self.pwm_neutral
        cmd.pitch = self.pwm_neutral
        cmd.roll = self.pwm_neutral

        # Visibility timeout
        if time.time() - self.last_seen_time > self.visibility_timeout:
            self.bucket_visible = False

        # -----------------------------
        # STATE MACHINE
        # -----------------------------
        if self.state == State.SEARCH:
            """
            Rotate until a bucket is detected.
            """
            cmd.yaw = self.search_yaw_pwm

            if self.bucket_visible:
                self.state = State.ALIGN

        elif self.state == State.ALIGN:
            """
            Center bucket horizontally.
            """
            if not self.bucket_visible:
                self.state = State.SEARCH
                return

            cmd.lateral = self.apply_p(self.bucket_x, self.kp_sway)

            if abs(self.bucket_x) < self.alignment_threshold:
                self.state = State.APPROACH

        elif self.state == State.APPROACH:
            """
            Move forward while correcting horizontal error.
            """
            if not self.bucket_visible:
                self.state = State.SEARCH
                return

            cmd.lateral = self.apply_p(self.bucket_x, self.kp_sway)
            cmd.forward = self.approach_forward_pwm

            if self.bucket_z < self.reach_distance:
                self.state = State.REACHED

        elif self.state == State.REACHED:
            """
            Stop all motion.
            """
            cmd.arm = 0
            cmd.forward = 0
            cmd.lateral = 0
            cmd.yaw = 0

        self.cmd_pub.publish(cmd)

    # -----------------------------
    # Helper
    # -----------------------------
    def apply_p(self, error, kp):
        """
        Proportional controller with saturation.
        """
        output = int(error * kp)
        output = max(min(output, self.pwm_range), -self.pwm_range)
        return self.pwm_neutral + output


def main(args=None):
    rclpy.init(args=args)
    node = BucketController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
