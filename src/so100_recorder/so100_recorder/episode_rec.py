import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from pynput import keyboard

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class SO100LeRobotRecorder(Node):
    def __init__(self):
        super().__init__('so100_lerobot_recorder')
        self.bridge = CvBridge()

        # 1. Setup LeRobot Dataset structure (Auto-Versioning)
        base_repo_id = "so100_teleop"
        self.dataset_path = os.path.expanduser(f"~/so100_dataset/{base_repo_id}")
        current_repo_id = base_repo_id
        counter = 1
        
        # Keep checking until we find a folder name that doesn't exist yet
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
        self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        self.create_subscription(Image, '/camera_overhead', self.overhead_cb, 10)
        self.create_subscription(Image, '/gripper_Cameras', self.wrist_cb, 10)
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

        # Episode & Toggle logic
        self.is_recording = False
        self.was_recording = False
        self.episode_frames = 0
        self.is_synced = False
        self._prev_space_state = False  

        # Keyboard state
        self.pressed_keys = set()
        self.step = 0.03

        self.get_logger().info("⌨️ Keyboard control active.")
        self.get_logger().info("Controls:")
        self.get_logger().info("  A/D -> shoulder_pan")
        self.get_logger().info("  W/S -> elbow_flex")
        self.get_logger().info("  I/K -> shoulder_lift")
        self.get_logger().info("  J/L -> wrist_flex")
        self.get_logger().info("  U/O -> wrist_roll")
        self.get_logger().info("  R/F -> gripper")
        self.get_logger().info("  SPACE -> start/stop recording")
        self.get_logger().info("  H -> reset arm to Home")

        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()

        self.timer = self.create_timer(1.0 / 30.0, self.main_loop)

    def state_callback(self, msg):
        if msg.name:
            for i, name in enumerate(msg.name):
                if name in self.current_angles and i < len(msg.position):
                    self.current_angles[name] = msg.position[i]

            if not self.is_synced:
                self.target_angles = self.current_angles.copy()
                self.is_synced = True
                self.get_logger().info("✅ Synced with robot state. Keyboard teleop ready.")

    def overhead_cb(self, msg):
        self.latest_overhead = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def wrist_cb(self, msg):
        self.latest_wrist = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def on_press(self, key):
        try:
            k = key.char.lower()
            self.pressed_keys.add(k)
        except AttributeError:
            if key == keyboard.Key.space:
                self.pressed_keys.add('space')

    def on_release(self, key):
        try:
            k = key.char.lower()
            self.pressed_keys.discard(k)
        except AttributeError:
            if key == keyboard.Key.space:
                self.pressed_keys.discard('space')

    def apply_keyboard_control(self):
        # Joint movement
        if 'a' in self.pressed_keys:
            self.target_angles['shoulder_pan'] += self.step
        if 'd' in self.pressed_keys:
            self.target_angles['shoulder_pan'] -= self.step

        if 'w' in self.pressed_keys:
            self.target_angles['elbow_flex'] -= self.step
        if 's' in self.pressed_keys:
            self.target_angles['elbow_flex'] += self.step

        if 'i' in self.pressed_keys:
            self.target_angles['shoulder_lift'] += self.step
        if 'k' in self.pressed_keys:
            self.target_angles['shoulder_lift'] -= self.step

        if 'j' in self.pressed_keys:
            self.target_angles['wrist_flex'] += self.step
        if 'l' in self.pressed_keys:
            self.target_angles['wrist_flex'] -= self.step

        if 'u' in self.pressed_keys:
            self.target_angles['wrist_roll'] += self.step
        if 'o' in self.pressed_keys:
            self.target_angles['wrist_roll'] -= self.step

        if 'r' in self.pressed_keys:
            self.target_angles['gripper'] += self.step
        if 'f' in self.pressed_keys:
            self.target_angles['gripper'] -= self.step

        # Reset to home
        if 'h' in self.pressed_keys:
            for name in self.joint_names:
                self.target_angles[name] = 0.0
            self.get_logger().info("🔄 Arm reset to original home position!")
            self.pressed_keys.discard('h')

        # Toggle recording with space
        current_space = 'space' in self.pressed_keys

        if current_space and not self._prev_space_state:
            self.is_recording = not self.is_recording
            if self.is_recording:
                self.get_logger().info("🔴 RECORDING STARTED: New Episode running...")
                self.episode_frames = 0
            else:
                if self.episode_frames > 0:
                    self.dataset.save_episode() # I added this back in!
                    self.get_logger().info(f"✅ EPISODE SAVED! ({self.episode_frames} frames stored).")

        self._prev_space_state = current_space

    def main_loop(self):
        if not self.is_synced:
            return

        # Apply keyboard control
        self.apply_keyboard_control()

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
            
        cv2.waitKey(1) # Critical for OpenCV to actually render the windows

        # 2. Add frame data if currently recording
        if self.is_recording and self.latest_overhead is not None and self.latest_wrist is not None:
            overhead_rgb = cv2.cvtColor(self.latest_overhead, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(self.latest_wrist, cv2.COLOR_BGR2RGB)

            # --- FORCE RESIZE TO 480p ---
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
                "task": "Pick up the yellow rope and place it in the red bowl"
            }

            self.dataset.add_frame(frame_data)
            self.episode_frames += 1

    def destroy_node(self):
        if hasattr(self, "keyboard_listener"):
            self.keyboard_listener.stop()
        cv2.destroyAllWindows() # Ensures windows close cleanly when you exit
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