#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import serial
import time
import os
import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class SO100HybridRecorder(Node):
    def __init__(self):
        super().__init__('so100_hybrid_recorder')

        self.get_logger().info("🚀 Starting 4-Axis Hardware + LeRobot Data Collector...")

        # --- 1. SETUP SERIAL HARDWARE (Direct pure-python bypass) ---
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.5)
            self.get_logger().info("✅ Serial Port Opened")
        except Exception as e:
            self.get_logger().error(f"❌ Could not open serial port: {e}")
            exit(1)

        # 🚨 4-AXIS ARM CONFIGURATION
        self.speed_multiplier = 15.0 
        self.limits = {
            1: [0,    2500],  # Base Pan
            2: [1698, 4090],  # Transplanted Shoulder Lift
            3: [460,  2677],  # Elbow Flex
            6: [3354, 4095]   # Gripper
        }
        self.home_positions = {
            1: 1250, 
            2: 2894,  
            3: 1568,  
            6: 3724   
        }
        self.last_sent_targets = {1: -1, 2: -1, 3: -1}
        self.motor_ids = [1, 2, 3]

        # Hardware Init
        self.targets = self.read_initial_positions()
        self.get_logger().info("🔒 Engaging motors...")
        for i in self.motor_ids:
            self.enable_torque(i, True)
        self.get_logger().info("✅ Torque enabled. Arm is locked in.")

        # --- 2. SETUP DIRECT CAMERA (OpenCV) ---
        # 0 is usually the built-in webcam. Change to 1 or 2 if using an external USB camera.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error("❌ Could not open USB camera!")
            exit(1)
        self.get_logger().info("📸 Direct Camera Feed Active.")

        # --- 3. SETUP LEROBOT DATASET ---
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
                "observation.state": {"dtype": "float32", "shape": (4,)},
                "action": {"dtype": "float32", "shape": (4,)},
                "observation.images.main_camera": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"]
                },
            }
        )

        # --- 4. SETUP ROS SUBSCRIPTIONS ---
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.joy_msg = None

        # State Variables
        self.is_recording = False
        self.episode_frames = 0
        self._prev_record_btn = 0

        # Timer matching the dataset FPS (30Hz = ~0.033s)
        self.timer = self.create_timer(1.0 / 30.0, self.main_loop)

    # --- HARDWARE METHODS ---
    def enable_torque(self, motor_id, enable=True):
        val = 1 if enable else 0
        packet = [0xFF, 0xFF, motor_id, 0x04, 0x03, 0x28, val]
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        self.ser.write(bytearray(packet))
        time.sleep(0.01)

    def read_initial_positions(self):
        targets = {}
        for motor_id in self.motor_ids:
            self.ser.reset_input_buffer()
            packet = [0xFF, 0xFF, motor_id, 0x04, 0x02, 0x38, 0x02]
            checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
            packet.append(checksum)
            
            self.ser.write(bytearray(packet))
            response = self.ser.read(8)
            
            if len(response) == 8 and response[0] == 0xFF and response[1] == 0xFF:
                pos = response[5] | (response[6] << 8)
                pos = max(self.limits[motor_id][0], min(pos, self.limits[motor_id][1]))
                targets[motor_id] = float(pos)
            else:
                targets[motor_id] = self.home_positions[motor_id]
            time.sleep(0.01)
        return targets

    def write_position(self, motor_id, position_ticks):
        safe_ticks = max(0, min(int(position_ticks), 4095))
        pos_low  = safe_ticks & 0xFF
        pos_high = (safe_ticks >> 8) & 0xFF
        packet = [0xFF, 0xFF, motor_id, 0x05, 0x03, 0x2A, pos_low, pos_high]
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        self.ser.write(bytearray(packet))

    # --- ROS CALLBACKS & LOOP ---
    def joy_callback(self, msg):
        self.joy_msg = msg

    def apply_joy_control(self):
        if self.joy_msg is None:
            return

        axes = self.joy_msg.axes
        buttons = self.joy_msg.buttons

        if len(axes) >= 8 and len(buttons) >= 6:
            # Map ROS Joy to our PS5 Logic
            # Axes: 6 is D-Pad Left/Right, 7 is D-Pad Up/Down
            dpad_x = axes[6] if len(axes) > 6 else 0.0
            dpad_y = axes[7] if len(axes) > 7 else 0.0
            
            btn_cross = buttons[0]
            btn_tri = buttons[2]
            btn_sq = buttons[3]  # Home Button
            btn_r1 = buttons[5]  # Record Button
            
            axis_l2 = axes[2]
            axis_r2 = axes[5]

            # --- RECORDING TOGGLE LOGIC ---
            if btn_r1 and not self._prev_record_btn:
                self.is_recording = not self.is_recording
                if self.is_recording:
                    self.get_logger().info("🔴 RECORDING STARTED: New Episode running...")
                    self.episode_frames = 0
                else:
                    if self.episode_frames > 0:
                        self.dataset.save_episode()
                        self.get_logger().info(f"✅ EPISODE SAVED! ({self.episode_frames} frames stored).")
            self._prev_record_btn = btn_r1

            # --- RETURN TO HOME LOGIC ---
            if btn_sq:
                for m in self.motor_ids:
                    if self.targets[m] < self.home_positions[m] - self.speed_multiplier:
                        self.targets[m] += self.speed_multiplier
                    elif self.targets[m] > self.home_positions[m] + self.speed_multiplier:
                        self.targets[m] -= self.speed_multiplier
                    else:
                        self.targets[m] = self.home_positions[m]
            else:
                # --- NORMAL TELEOP LOGIC ---
                self.targets[1] -= dpad_x * self.speed_multiplier # Inverted for correct L/R mapping
                self.targets[2] += dpad_y * self.speed_multiplier

                if btn_tri: self.targets[3] += self.speed_multiplier
                if btn_cross: self.targets[3] -= self.speed_multiplier

                # L2/R2 mapped -1.0 to 1.0 in ROS joy
                if axis_l2 < -0.5: self.targets[6] += (self.speed_multiplier * 0.5)
                if axis_r2 < -0.5: self.targets[6] -= (self.speed_multiplier * 0.5)

    def main_loop(self):
        # 1. Update targets based on controller
        self.apply_joy_control()

        # 2. Send commands to hardware
        for motor_id in self.motor_ids:
            lo, hi = self.limits[motor_id]
            self.targets[motor_id] = max(lo, min(self.targets[motor_id], hi))
            
            if self.targets[motor_id] != self.last_sent_targets[motor_id]:
                self.write_position(motor_id, self.targets[motor_id])
                self.last_sent_targets[motor_id] = self.targets[motor_id]

        # 3. Read direct camera frame
        ret, frame = self.cap.read()
        if not ret:
            return

        # 4. Add frame to Dataset if recording
        if self.is_recording:
            # Convert BGR to RGB for LeRobot
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Arrays must strictly match the 4-axis schema
            current_state = np.array([self.targets[m] for m in self.motor_ids], dtype=np.float32)
            
            frame_data = {
                "observation.state": current_state,
                "action": current_state,  # Assuming perfect execution for kinesthetic logging
                "observation.images.main_camera": frame_rgb,
                "task": "pick up the box and place it in cylinder"
            }

            self.dataset.add_frame(frame_data)
            self.episode_frames += 1

        # 5. Visualizer
        preview = cv2.resize(frame, (640, 480))
        status_text = "RECORDING" if self.is_recording else "IDLE (R1 to Start)"
        color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Main Camera", preview)
        cv2.waitKey(1)

    def destroy_node(self):
        self.get_logger().info("🔓 Releasing motors and camera...")
        for i in self.motor_ids:
            self.enable_torque(i, False)
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SO100HybridRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()