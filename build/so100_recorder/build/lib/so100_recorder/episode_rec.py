import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, Joy
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from rclpy.qos import qos_profile_sensor_data

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class SO100LeRobotRecorder(Node):
    def __init__(self):
        super().__init__('so100_lerobot_recorder')
        self.bridge = CvBridge()

        # 1. Setup LeRobot Dataset structure
        base_repo_id = "so100_teleop"
        self.dataset_path = os.path.expanduser(f"~/so100_dataset/{base_repo_id}")
        current_repo_id = base_repo_id
        counter = 1
        
        while os.path.exists(self.dataset_path):
            current_repo_id = f"{base_repo_id}_{counter}"
            self.dataset_path = os.path.expanduser(f"~/so100_dataset/{current_repo_id}")
            counter += 1

        self.get_logger().info(f"✨ Creating brand new dataset at: {current_repo_id}")
        
        self.dataset = LeRobotDataset.create(
            repo_id=current_repo_id,
            fps=30,
            root=self.dataset_path,
            features={
                "observation.state": {"dtype": "float32", "shape": (6,)},
                "action": {"dtype": "float32", "shape": (6,)},
                "observation.images.overhead": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"]
                },
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"]
                },
            }
        )

        # 2. Setup ROS 2 Topics
        self.create_subscription(JointState, '/joint_states', self.state_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera_overhead', self.overhead_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/gripper_Cameras', self.wrist_cb, qos_profile_sensor_data)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)

        # 3. State Variables
        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
            'gripper'
        ]
        self.current_angles = {name: 0.0 for name in self.joint_names}
        self.target_angles = {name: 0.0 for name in self.joint_names}

        self.latest_overhead = None
        self.latest_wrist = None
        self.joy_msg = None

        # Episode & Toggle logic
        self.is_recording = False
        self.episode_frames = 0
        self.is_synced = False
        
        # Debounce tracking for controller buttons
        self._prev_record_btn = 0
        self._prev_home_btn = 0  

        self.step = 0.03 # Max speed multiplier

        self.get_logger().info("🎮 Custom PS5 Controller Layout Active.")
        self.get_logger().info("Controls:")
        self.get_logger().info("  Left Stick (L/R) -> Shoulder Pan")
        self.get_logger().info("  Left Stick (U/D) -> Shoulder Lift")
        self.get_logger().info("  Right Stick (U/D) -> Elbow Flex")
        self.get_logger().info("  Right Stick (L/R) -> Wrist Flex")
        self.get_logger().info("  L1/R1 -> Gripper Close/Open")
        self.get_logger().info("  L2/R2 -> Wrist Roll Left/Right")
        self.get_logger().info("  X Button -> Start/Stop recording")
        self.get_logger().info("  Triangle -> Reset arm to Home")

        self.timer = self.create_timer(1.0 / 30.0, self.main_loop)

    def state_callback(self, msg):
        if msg.name:
            for i, name in enumerate(msg.name):
                if name in self.current_angles and i < len(msg.position):
                    self.current_angles[name] = msg.position[i]

            if not self.is_synced:
                self.target_angles = self.current_angles.copy()
                self.is_synced = True
                self.get_logger().info("✅ Synced with robot state. PS5 teleop ready.")

    def overhead_cb(self, msg):
        self.latest_overhead = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def wrist_cb(self, msg):
        self.latest_wrist = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def joy_callback(self, msg):
        self.joy_msg = msg

    def apply_joy_control(self):
        if self.joy_msg is None:
            return

        axes = self.joy_msg.axes
        buttons = self.joy_msg.buttons

        # Ensure arrays are long enough to avoid indexing errors
        if len(axes) >= 5 and len(buttons) >= 8:
            
            # --- THE 4 ESSENTIAL STICK MOVEMENTS ---
            # Left Stick
            self.target_angles['shoulder_pan'] += axes[0] * self.step
            self.target_angles['shoulder_lift'] += axes[1] * self.step
            
            # Right Stick
            self.target_angles['wrist_flex'] += axes[3] * self.step
            self.target_angles['elbow_flex'] -= axes[4] * self.step

            # --- TRIGGERS / BUMPERS ---
            # Gripper control (L1 = Close, R1 = Open)
            if buttons[4]: # L1
                self.target_angles['gripper'] -= self.step
            if buttons[5]: # R1
                self.target_angles['gripper'] += self.step

            # Wrist Roll (L2 = Roll Left, R2 = Roll Right)
            if buttons[6]: # L2
                self.target_angles['wrist_roll'] += self.step
            if buttons[7]: # R2
                self.target_angles['wrist_roll'] -= self.step

            # --- UTILITY BUTTONS ---
            # Reset to home (Triangle button = index 2)
            current_home_btn = buttons[2]
            if current_home_btn and not self._prev_home_btn:
                for name in self.joint_names:
                    self.target_angles[name] = 0.0
                self.get_logger().info("🔄 Arm reset to original home position!")
            self._prev_home_btn = current_home_btn

            # Toggle recording (X button = index 0)
            current_record_btn = buttons[0]
            if current_record_btn and not self._prev_record_btn:
                self.is_recording = not self.is_recording
                if self.is_recording:
                    self.get_logger().info("🔴 RECORDING STARTED: New Episode running...")
                    self.episode_frames = 0
                else:
                    if self.episode_frames > 0:
                        self.dataset.save_episode()
                        self.get_logger().info(f"✅ EPISODE SAVED! ({self.episode_frames} frames stored).")
            self._prev_record_btn = current_record_btn

    def main_loop(self):
        if not self.is_synced:
            return

        # Apply controller input
        self.apply_joy_control()

        # 1. Command the robot
        cmd = JointState()
        cmd.name = self.joint_names
        cmd.position = [self.target_angles[n] for n in self.joint_names]
        self.cmd_pub.publish(cmd)

        # --- LIVE CAMERA PREVIEW ---
        if self.latest_overhead is not None:
            preview_overhead = cv2.resize(self.latest_overhead, (640, 480))
            cv2.imshow("Overhead Camera", preview_overhead)
            
        if self.latest_wrist is not None:
            preview_wrist = cv2.resize(self.latest_wrist, (640, 480))
            cv2.imshow("Wrist Camera", preview_wrist)
            
        cv2.waitKey(1)

        # 2. Add frame data if currently recording
        if self.is_recording and self.latest_overhead is not None and self.latest_wrist is not None:
            overhead_rgb = cv2.cvtColor(self.latest_overhead, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(self.latest_wrist, cv2.COLOR_BGR2RGB)

            overhead_rgb = cv2.resize(overhead_rgb, (640, 480))
            wrist_rgb = cv2.resize(wrist_rgb, (640, 480))

            frame_data = {
                "observation.state": np.array(
                    [self.current_angles[n] for n in self.joint_names],
                    dtype=np.float32
                ),
                "action": np.array(
                    [self.target_angles[n] for n in self.joint_names],
                    dtype=np.float32
                ),
                "observation.images.overhead": overhead_rgb,
                "observation.images.wrist": wrist_rgb,
                "task": "Pick up the rope and place it in the bowl"
            }

            self.dataset.add_frame(frame_data)
            self.episode_frames += 1

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = SO100LeRobotRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()