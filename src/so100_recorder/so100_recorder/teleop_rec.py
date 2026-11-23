#!/usr/bin/env python3
"""PS5 teleoperation of the real 4-axis SO-100 arm with LeRobot dataset recording."""
import time

import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import serial

from so100_recorder import config, feetech

TASK_DESCRIPTION = 'pick up the box and place it in cylinder'

# Ticks moved per control tick while a button is held.
SPEED_MULTIPLIER = 15.0

# Joy mapping (ROS joy driver, PS5 controller).
BTN_CROSS = 0
BTN_TRIANGLE = 2
BTN_SQUARE = 3   # Return to home
BTN_R1 = 5       # Toggle recording
AXIS_L2 = 2
AXIS_R2 = 5
AXIS_DPAD_X = 6
AXIS_DPAD_Y = 7


class SO100HybridRecorder(Node):
    def __init__(self):
        super().__init__('so100_hybrid_recorder')
        self.get_logger().info('Starting 4-axis hardware + LeRobot data collector...')

        # --- Serial hardware ---
        try:
            self.ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.5)
            self.get_logger().info('Serial port opened')
        except Exception as e:
            self.get_logger().error(f'Could not open serial port: {e}')
            raise SystemExit(1)

        self.limits = config.TICK_LIMITS
        self.home_positions = config.HOME_POSITIONS
        # Only these motors are read, commanded and recorded (the gripper, ID 6, is not).
        self.motor_ids = [1, 2, 3]
        self.last_sent_targets = {m: -1 for m in self.motor_ids}

        self.targets = self.read_initial_positions()
        self.get_logger().info('Engaging motors...')
        for motor_id in self.motor_ids:
            self.enable_torque(motor_id, True)
        self.get_logger().info('Torque enabled. Arm is locked in.')

        # --- Camera ---
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open USB camera!')
            raise SystemExit(1)
        self.get_logger().info('Direct camera feed active.')

        # --- LeRobot dataset ---
        repo_id, self.dataset_path = config.next_free_dataset_path()
        self.get_logger().info(f'Creating new dataset at: {repo_id}')
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=config.DATASET_FPS,
            root=self.dataset_path,
            features={
                'observation.state': {'dtype': 'float32', 'shape': (4,)},
                'action': {'dtype': 'float32', 'shape': (4,)},
                'observation.images.main_camera': {
                    'dtype': 'video',
                    'shape': (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3),
                    'names': ['height', 'width', 'channels'],
                },
            },
        )

        # --- ROS ---
        self.joy_msg = None
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.is_recording = False
        self.episode_frames = 0
        self._prev_record_btn = 0

        self.timer = self.create_timer(1.0 / config.DATASET_FPS, self.main_loop)

    # --- Hardware ---
    def enable_torque(self, motor_id, enable=True):
        feetech.enable_torque(self.ser, motor_id, enable)
        time.sleep(0.01)

    def read_initial_positions(self):
        targets = {}
        for motor_id in self.motor_ids:
            pos = feetech.read_position(self.ser, motor_id)
            if pos is not None:
                lo, hi = self.limits[motor_id]
                targets[motor_id] = float(max(lo, min(pos, hi)))
            else:
                targets[motor_id] = self.home_positions[motor_id]
            time.sleep(0.01)
        return targets

    def write_position(self, motor_id, position_ticks):
        feetech.write_position(self.ser, motor_id, position_ticks)

    # --- Control ---
    def joy_callback(self, msg):
        self.joy_msg = msg

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.get_logger().info('RECORDING STARTED: new episode running...')
            self.episode_frames = 0
        elif self.episode_frames > 0:
            self.dataset.save_episode()
            self.get_logger().info(f'EPISODE SAVED! ({self.episode_frames} frames stored).')

    def step_toward_home(self):
        step = SPEED_MULTIPLIER
        for m in self.motor_ids:
            home = self.home_positions[m]
            if self.targets[m] < home - step:
                self.targets[m] += step
            elif self.targets[m] > home + step:
                self.targets[m] -= step
            else:
                self.targets[m] = home

    def apply_joy_control(self):
        if self.joy_msg is None:
            return

        axes = self.joy_msg.axes
        buttons = self.joy_msg.buttons
        if len(axes) < 8 or len(buttons) < 6:
            return

        btn_r1 = buttons[BTN_R1]
        if btn_r1 and not self._prev_record_btn:
            self.toggle_recording()
        self._prev_record_btn = btn_r1

        if buttons[BTN_SQUARE]:
            self.step_toward_home()
            return

        step = SPEED_MULTIPLIER
        self.targets[1] -= axes[AXIS_DPAD_X] * step  # Inverted for correct L/R mapping
        self.targets[2] += axes[AXIS_DPAD_Y] * step

        if buttons[BTN_TRIANGLE]:
            self.targets[3] += step
        if buttons[BTN_CROSS]:
            self.targets[3] -= step

        # Triggers rest at 1.0 and go to -1.0 when fully pressed.
        if axes[AXIS_L2] < -0.5:
            self.targets[6] += step * 0.5
        if axes[AXIS_R2] < -0.5:
            self.targets[6] -= step * 0.5

    def main_loop(self):
        self.apply_joy_control()

        # Send only targets that changed.
        for motor_id in self.motor_ids:
            lo, hi = self.limits[motor_id]
            self.targets[motor_id] = max(lo, min(self.targets[motor_id], hi))
            if self.targets[motor_id] != self.last_sent_targets[motor_id]:
                self.write_position(motor_id, self.targets[motor_id])
                self.last_sent_targets[motor_id] = self.targets[motor_id]

        ret, frame = self.cap.read()
        if not ret:
            return

        if self.is_recording:
            current_state = np.array([self.targets[m] for m in self.motor_ids], dtype=np.float32)
            self.dataset.add_frame({
                'observation.state': current_state,
                # Commanded target is logged as both state and action (assumes perfect tracking).
                'action': current_state,
                'observation.images.main_camera': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                'task': TASK_DESCRIPTION,
            })
            self.episode_frames += 1

        preview = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        status_text = 'RECORDING' if self.is_recording else 'IDLE (R1 to Start)'
        color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Main Camera', preview)
        cv2.waitKey(1)

    def destroy_node(self):
        self.get_logger().info('Releasing motors and camera...')
        for motor_id in self.motor_ids:
            self.enable_torque(motor_id, False)
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
        node.get_logger().info('Recording manually stopped.')
    except Exception as e:
        node.get_logger().error(f'Fatal error during recording: {e}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
