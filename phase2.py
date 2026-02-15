#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from custom_msgs.msg import Commands, Telemetry
import math
import time
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class State:
	SEARCH = 0  # surges forward to see bucket in down cam
	ALIGN_XY = 1 # align with bucket, need to calculate camera to dropper dist
	APPROACH = 3  #  towards the bucket
	LOCK = 4  # lock it in place to drop ball
	#maybe some sort of check or redo im not sure

class bucketControls(Node):
    def __init__(self):
        super().__init__("bucket_control2")

        # --- Tunable PID Parameters ---
        self.kp_sway = 200.0  # PWM counts per meter error
        self.kp_surge = 150.0 
        self.kp_yaw = 300.0   # PWM counts per radian error
        
        # --- Config ---
        self.pwm_neutral = 1500
        self.pwm_range = 400 # +/- 400 from neutral
        self.search_surge_pwm = 1600 # Approx 0.3 m/s
        self.lower_to_bucket = #how long??
        
        # State Variables
        self.state = State.SEARCH
        self.last_pose_time = 0.0
        self.bucket_visible = False
        self.target_pose = None # [x, y, z, roll, pitch, yaw] relative to camera
        self.current_heading = 0.0
        self.blind_timer_start = None

        #ROS
        self.cmd_pub = self.create_publisher(Commands, "/master/commands", 10)
        
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)
self.create_subscription(PoseStamped, "ellipsoid_path", self.pose_callback, qos_profile)
        self.create_subscription(Telemetry, "/master/telemetry", self.telem_callback, 10)

        #Loop
        self.create_timer(0.05, self.control_loop) #should time be longer or shorter idk
       
    # Callbacks

    def telem_callback(self, msg):
        self.current_heading = msg.heading

    def pose_callback(self, msg):
        self.last_pose_time = time.time()
        self.bucket_visible = True
        
       ''' # Convert Quaternion to Euler (Yaw specifically)
        q = msg.pose.orientation
        (roll, pitch, yaw) = self.euler_from_quaternion(q.x, q.y, q.z, q.w)'''
        

        # Store relative pose: x (right), y (down), z (forward) - Standard Camera Frame
        # Adjust depending on your specific camera frame setup!
        self.target_pose = {
            'x': msg.pose.position.x, # Horizontal Error
            'y': msg.pose.position.y, # Vertical Error (Depth diff)
            'z': msg.pose.position.z, # Distance
            #'yaw': yaw
        }

    # --- Main Loop ---

    def control_loop(self):
        cmd = Commands()
        cmd.mode = "ALT_HOLD" # Ensure we are in depth hold mode
        cmd.arm = 1
        
        # Reset channels to neutral
        cmd.pitch = 1500
        cmd.roll = 1500
        cmd.thrust = 1500
        cmd.yaw = 1500
        cmd.forward = 1500
        cmd.lateral = 1500

        # Check visibility timeout (Lost marker)
        if time.time() - self.last_pose_time > 1.0:
            self.bucket_visible = False

        # --- STATE MACHINE ---
        if self.state == State.SEARCH:
            # Phase 1: Search
            cmd.forward = self.search_surge_pwm # Surge 0.3
            self.get_logger().info("Phase 1: Searching...", throttle_duration_sec=1)
            
            if self.dock_visible:
                self.get_logger().info("bucket found -> align xy")
                self.state = State.ALIGN_XY
	 elif self.state == State.ALIGN_XY:

   	 if not self.bucket_visible:
       	 self.state = State.SEARCH
       	 return

    # Errors from downward camera frame
    err_x = self.target_pose['x']   # left/right
    err_y = self.target_pose['y']   # forward/back

    # Center over bucket
    cmd.lateral = self.apply_pid(err_x, self.kp_sway)
    cmd.forward = self.apply_pid(err_y, self.kp_surge)

    # Hold height for now
    cmd.thrust = 1500

    # Check if centered over bucket
    if abs(err_x) < 0.1 and abs(err_y) < 0.1:
        self.get_logger().info("XY aligned over bucket. Beginning descent.")
        self.state = State.APPROACH


	# Descend
	elif self.state == State.APPROACH:

    	# If bucket disappears while very close 
    if not self.bucket_visible and self.target_pose['z'] < 0.5:
        self.get_logger().info("Bucket lost at close range — locking position")
        self.blind_timer_start = time.time()
        self.state = State.LOCK
        return

    elif not self.bucket_visible:
        self.state = State.SEARCH
        return

    # Keep correcting horizontal drift while descending
    err_x = self.target_pose['x']
    err_y = self.target_pose['y']

    cmd.lateral = self.apply_pid(err_x, self.kp_sway)
    cmd.forward = self.apply_pid(err_y, self.kp_surge)

    # Controlled descent
    if self.target_pose['z'] > 1.0:
        cmd.thrust = 1560   # faster descent when high
    else:
        cmd.thrust = 1530   # slow near bucket


# lock in place
elif self.state == State.LOCK:

    elapsed = time.time() - self.blind_timer_start

    # Stop all motion and stabilize
    cmd.forward = 1500
    cmd.lateral = 1500
    cmd.thrust = 1500
    cmd.yaw = 1500

    if elapsed > 3.0:
        self.get_logger().info("Position locked over bucket")

        self.cmd_pub.publish(cmd)

    # --- Helpers ---
    
    def apply_pid(self, error, kp):
        output = int(error * kp)
        # Clamp
        output = max(min(output, self.pwm_range), -self.pwm_range)
        return self.pwm_neutral + output

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z 

def main(args=None):
    rclpy.init(args=args)
    node = BucketControls()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
