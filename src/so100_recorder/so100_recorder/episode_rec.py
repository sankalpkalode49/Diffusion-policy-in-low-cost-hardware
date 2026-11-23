"""PS5 teleoperation of the simulated 6-DOF SO-100 (Isaac Sim) with two-camera LeRobot recording."""
import cv2
from cv_bridge import CvBridge
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState, Joy

from so100_recorder import config

TASK_DESCRIPTION = 'Pick up the rope and place it in the bowl'

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

# Radians moved per control tick at full stick deflection.
STEP = 0.03

# Joy mapping (ROS joy driver, PS5 controller).
BTN_CROSS = 0     # Toggle recording
BTN_TRIANGLE = 2  # Reset to home
BTN_L1 = 4
BTN_R1 = 5
BTN_L2 = 6
BTN_R2 = 7

PREVIEW_SIZE = (480, 360)


class SO100LeRobotRecorder(Node):
    def __init__(self):
        super().__init__('so100_lerobot_recorder')
        self.bridge = CvBridge()

        # --- LeRobot dataset ---
        repo_id, self.dataset_path = config.next_free_dataset_path()
        self.get_logger().info(f'Creating new dataset at: {repo_id}')
        image_feature = {
            'dtype': 'video',
            'shape': (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3),
            'names': ['height', 'width', 'channels'],
        }
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=config.DATASET_FPS,
            root=self.dataset_path,
            features={
                'observation.state': {'dtype': 'float32', 'shape': (len(JOINT_NAMES),)},
                'action': {'dtype': 'float32', 'shape': (len(JOINT_NAMES),)},
                'observation.images.overhead': dict(image_feature),
                'observation.images.wrist': dict(image_feature),
            },
        )

        # --- ROS topics (best-effort QoS for cameras to reduce latency) ---
        self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        self.create_subscription(Image, '/camera_overhead', self.overhead_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/gripper_Cameras', self.wrist_cb, qos_profile_sensor_data)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)

        # --- State ---
        self.current_angles = {name: 0.0 for name in JOINT_NAMES}
        self.target_angles = {name: 0.0 for name in JOINT_NAMES}
        self.latest_overhead = None
        self.latest_wrist = None
        self.joy_msg = None

        self.is_recording = False
        self.episode_frames = 0
        self.is_synced = False

        # Previous button states, for edge detection.
        self._prev_record_btn = 0
        self._prev_home_btn = 0

        self.get_logger().info('PS5 controller layout:')
        self.get_logger().info('  Left stick (L/R)  -> Shoulder pan')
        self.get_logger().info('  Left stick (U/D)  -> Shoulder lift')
        self.get_logger().info('  Right stick (U/D) -> Elbow flex (reversed for SO100)')
        self.get_logger().info('  Right stick (L/R) -> Wrist flex')
        self.get_logger().info('  L1/R1             -> Gripper close/open')
        self.get_logger().info('  L2/R2             -> Wrist roll left/right')
        self.get_logger().info('  X                 -> Start/stop recording')
        self.get_logger().info('  Triangle          -> Reset arm to home')

        self.timer = self.create_timer(1.0 / config.DATASET_FPS, self.main_loop)

    # --- Callbacks ---
    def state_callback(self, msg):
        if not msg.name:
            return
        for i, name in enumerate(msg.name):
            if name in self.current_angles and i < len(msg.position):
                self.current_angles[name] = msg.position[i]

        if not self.is_synced:
            self.target_angles = self.current_angles.copy()
            self.is_synced = True
            self.get_logger().info('Synced with robot state. PS5 teleop ready.')

    def overhead_cb(self, msg):
        self.latest_overhead = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def wrist_cb(self, msg):
        self.latest_wrist = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def joy_callback(self, msg):
        self.joy_msg = msg

    # --- Control ---
    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.get_logger().info('RECORDING STARTED: new episode running...')
            self.episode_frames = 0
        elif self.episode_frames > 0:
            self.dataset.save_episode()
            self.get_logger().info(f'EPISODE SAVED! ({self.episode_frames} frames stored).')

    def apply_joy_control(self):
        if self.joy_msg is None:
            return

        axes = self.joy_msg.axes
        buttons = self.joy_msg.buttons
        if len(axes) < 5 or len(buttons) < 8:
            return

        # Sticks
        self.target_angles['shoulder_pan'] += axes[0] * STEP
        self.target_angles['shoulder_lift'] += axes[1] * STEP
        self.target_angles['wrist_flex'] += axes[3] * STEP
        # Elbow is inverted: the SO100 elbow servo is mounted upside down.
        self.target_angles['elbow_flex'] -= axes[4] * STEP

        # Bumpers / triggers
        if buttons[BTN_L1]:
            self.target_angles['gripper'] -= STEP
        if buttons[BTN_R1]:
            self.target_angles['gripper'] += STEP
        if buttons[BTN_L2]:
            self.target_angles['wrist_roll'] += STEP
        if buttons[BTN_R2]:
            self.target_angles['wrist_roll'] -= STEP

        home_btn = buttons[BTN_TRIANGLE]
        if home_btn and not self._prev_home_btn:
            for name in JOINT_NAMES:
                self.target_angles[name] = 0.0
            self.get_logger().info('Arm reset to home position.')
        self._prev_home_btn = home_btn

        record_btn = buttons[BTN_CROSS]
        if record_btn and not self._prev_record_btn:
            self.toggle_recording()
        self._prev_record_btn = record_btn

    def main_loop(self):
        if not self.is_synced:
            return

        self.apply_joy_control()

        cmd = JointState()
        cmd.name = JOINT_NAMES
        cmd.position = [self.target_angles[n] for n in JOINT_NAMES]
        self.cmd_pub.publish(cmd)

        if self.latest_overhead is None or self.latest_wrist is None:
            return

        if self.is_recording:
            size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
            self.dataset.add_frame({
                'observation.state': np.array([self.current_angles[n] for n in JOINT_NAMES], dtype=np.float32),
                'action': np.array([self.target_angles[n] for n in JOINT_NAMES], dtype=np.float32),
                'observation.images.overhead': cv2.resize(
                    cv2.cvtColor(self.latest_overhead, cv2.COLOR_BGR2RGB), size),
                'observation.images.wrist': cv2.resize(
                    cv2.cvtColor(self.latest_wrist, cv2.COLOR_BGR2RGB), size),
                'task': TASK_DESCRIPTION,
            })
            self.episode_frames += 1

        status_text = 'RECORDING' if self.is_recording else 'IDLE (X to Start)'
        color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        for window, image in (('Overhead Camera', self.latest_overhead), ('Wrist Camera', self.latest_wrist)):
            preview = cv2.resize(image, PREVIEW_SIZE)
            cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(window, preview)
        cv2.waitKey(1)


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
