import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from pynput import keyboard


class PS5TeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)
        self.state_sub = self.create_subscription(
            JointState, '/joint_states', self.state_callback, 10
        )

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
            'gripper'
        ]
        self.target_angles = {name: 0.0 for name in self.joint_names}

        self.is_synced = False
        self.step = 0.03

        self.pressed_keys = set()

        self.get_logger().info("⌨️ Waiting to sync with Isaac Sim...")
        self.get_logger().info("Controls:")
        self.get_logger().info("  A/D -> shoulder_pan")
        self.get_logger().info("  W/S -> elbow_flex")
        self.get_logger().info("  I/K -> shoulder_lift")
        self.get_logger().info("  J/L -> wrist_flex")
        self.get_logger().info("  U/O -> wrist_roll")
        self.get_logger().info("  R/F -> gripper")
        self.get_logger().info("  SPACE -> reset")

        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()

    def state_callback(self, msg):
        if not self.is_synced and msg.name:
            for i, name in enumerate(msg.name):
                if name in self.target_angles and i < len(msg.position):
                    self.target_angles[name] = msg.position[i]
            self.is_synced = True
            self.get_logger().info("✅ Synced! Keyboard control active.")

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.space:
                self.pressed_keys.add('space')

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.space:
                self.pressed_keys.discard('space')

    def timer_callback(self):
        if not self.is_synced:
            return

        # Joint controls
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

        if 'space' in self.pressed_keys:
            for name in self.joint_names:
                self.target_angles[name] = 0.0
            self.get_logger().info("🔄 Arm reset to original home position!")
            self.pressed_keys.discard('space')

        out_msg = JointState()
        out_msg.header = Header()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.name = self.joint_names
        out_msg.position = [self.target_angles[name] for name in self.joint_names]

        self.cmd_pub.publish(out_msg)

    def destroy_node(self):
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PS5TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()