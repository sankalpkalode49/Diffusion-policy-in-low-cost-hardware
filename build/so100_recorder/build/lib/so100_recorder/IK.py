#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import serial
import time
import os

class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')
        self.get_logger().info("🔴 Trajectory Recorder Active. Press R1 to start recording, R1 again to stop.")

        try:
            self.ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.5)
        except Exception as e:
            self.get_logger().error(f"Serial error: {e}")
            exit(1)

        self.motor_ids = [2, 3, 6] # Active motors (Shoulder, Elbow, Gripper)
        self.limits = {2: [1698, 4090], 3: [460,  2677], 6: [3354, 4095]}
        self.speed_multiplier = 15.0
        
        self.targets = {2: 2894.0, 3: 1568.0, 6: 3724.0}
        self.last_sent = {2: -1, 3: -1, 6: -1}

        # Structure-holding torque on the dead base motor (Motor 1)
        self.enable_torque(1, True)
        for i in self.motor_ids:
            self.enable_torque(i, True)

        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.joy_msg = None
        self.is_recording = False
        self.recorded_lines = []
        self._prev_r1 = 0

        self.timer = self.create_timer(1.0 / 30.0, self.main_loop) # Record at 30Hz

    def enable_torque(self, motor_id, enable=True):
        val = 1 if enable else 0
        packet = [0xFF, 0xFF, motor_id, 0x04, 0x03, 0x28, val]
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        self.ser.write(bytearray(packet))
        time.sleep(0.01)

    def write_position(self, motor_id, position_ticks):
        safe_ticks = max(self.limits[motor_id][0], min(int(position_ticks), self.limits[motor_id][1]))
        pos_low  = safe_ticks & 0xFF
        pos_high = (safe_ticks >> 8) & 0xFF
        packet = [0xFF, 0xFF, motor_id, 0x05, 0x03, 0x2A, pos_low, pos_high]
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        self.ser.write(bytearray(packet))

    def joy_callback(self, msg):
        self.joy_msg = msg

    def main_loop(self):
        if self.joy_msg is None: return
        axes = self.joy_msg.axes
        buttons = self.joy_msg.buttons

        if len(axes) >= 8 and len(buttons) >= 6:
            dpad_y = axes[7]
            btn_cross = buttons[0]
            btn_tri = buttons[2]
            btn_r1 = buttons[5]
            axis_l2 = axes[2]
            axis_r2 = axes[5]

            # Toggle Recording on R1 Press
            if btn_r1 and not self._prev_r1:
                self.is_recording = not self.is_recording
                if self.is_recording:
                    self.get_logger().info("🔴 RECORDING STARTED... Move the arm now!")
                    self.recorded_lines = []
                else:
                    self.get_logger().info("✅ RECORDING STOPPED. Saving trajectory.txt...")
                    with open(os.path.expanduser('~/so100_ws/trajectory.txt'), 'w') as f:
                        f.writelines(self.recorded_lines)
                    self.get_logger().info("🎉 Saved successfully to ~/so100_ws/trajectory.txt")
            self._prev_r1 = btn_r1

            # Teleop Control
            self.targets[2] += dpad_y * self.speed_multiplier
            if btn_tri: self.targets[3] += self.speed_multiplier
            if btn_cross: self.targets[3] -= self.speed_multiplier
            if axis_l2 < -0.5: self.targets[6] += (self.speed_multiplier * 0.5)
            if axis_r2 < -0.5: self.targets[6] -= (self.speed_multiplier * 0.5)

            # Send to hardware & log if recording
            for m in self.motor_ids:
                self.targets[m] = max(self.limits[m][0], min(self.targets[m], self.limits[m][1]))
                if self.targets[m] != self.last_sent[m]:
                    self.write_position(m, self.targets[m])
                    self.last_sent[m] = self.targets[m]

            if self.is_recording:
                # Store exactly what was sent: shoulder, elbow, gripper
                self.recorded_lines.append(f"{int(self.targets[2])},{int(self.targets[3])},{int(self.targets[6])}\n")

    def destroy_node(self):
        for i in [1, 2, 3, 6]: self.enable_torque(i, False)
        self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()