#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import serial
import time
import sys

class SO100InverseKinematicsNode(Node):
    def __init__(self):
        super().__init__('ik_deploy_node')
        self.get_logger().info("🚀 Initializing 3-Axis Calibrated Inverse Kinematics Controller...")

        # ==========================================
        # 🚨 HARDWARE TUNING & CALIBRATION OFFSETS
        # ==========================================
        self.SHOULDER_DIRECTION = 1  
        self.ELBOW_DIRECTION = -1     # Inverted for physically reversed elbow servo

        # 🔧 CALIBRATION OFFSET FIX: Tune this value based on your physical horizontal alignment.
        # If your arm sits -21.4° below true horizontal when at 2894 ticks, set this to -21.4
        self.SHOULDER_HOME_ANGLE_DEG = -95.6  

        # Hardware joint boundaries
        self.limits = {
            2: [1698, 4090],  # Shoulder Lift
            3: [460,  2677],  # Elbow Flex
            6: [3354, 4095]   # Gripper
        }
        self.home_positions = {
            2: 2894,  # Calibrated home tick baseline
            3: 1568,  # Calibrated home tick baseline
            6: 3724   # Gripper open home
        }

        self.TICKS_PER_DEGREE = 4095.0 / 360.0

        # Physical link measurements (in cm)
        self.L1 = 12.0          # Shoulder pivot to elbow pivot
        self.L2 = 25.0          # Elbow pivot to gripper tip
        self.base_height = 13.0 # Table surface to shoulder pivot center

        try:
            self.ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.1)
            self.get_logger().info("✅ Serial Port Opened Successfully")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to interface with serial port: {e}")
            exit(1)

        # Power up structural locks so the burnt base motor doesn't slide/sag
        self.get_logger().info("🔒 Engaging structural motor locks...")
        for i in [1, 2, 3, 6]:
            self.enable_torque(i, True)

        # ==========================================
        # 2. RUN DEMO SEQUENCE
        # ==========================================
        demo_sequence = [
            {"name": "Hover Over Box",     "x": 16.0, "z": 8.0,  "gripper": 4000},
            {"name": "Lower to Grasp Box", "x": 16.0, "z": 0.5,  "gripper": 4000},
            {"name": "Close Gripper",      "x": 16.0, "z": 0.5,  "gripper": 3400},
            {"name": "Lift Box Clear",     "x": 16.0, "z": 8.0,  "gripper": 3400},
            {"name": "Move to Cylinder",   "x": 22.0, "z": 8.0,  "gripper": 3400},
            {"name": "Lower Into Cylinder","x": 22.0, "z": 2.0,  "gripper": 3400},
            {"name": "Release Grasp",      "x": 22.0, "z": 2.0,  "gripper": 4000},
            {"name": "Retract Home",       "x": 16.0, "z": 12.0, "gripper": 4000}
        ]

        # PRE-FLIGHT MATHEMATICAL DRY-RUN
        self.get_logger().info("🔍 Running Mathematical Dry-Run...")
        math_failed = False
        for step in demo_sequence:
            s, e = self.calculate_ik(step['x'], step['z'])
            if s is None:
                self.get_logger().error(f"❌ UNREACHABLE: {step['name']} at x={step['x']}, z={step['z']}")
                math_failed = True
            else:
                self.get_logger().info(f"✅ {step['name']}: shoulder={s:.1f}°, elbow={e:.1f}°")
        
        if math_failed:
            self.get_logger().error("🛑 Math check failed. Target out of bounds. Aborting execution loop.")
            self.destroy_node()
            sys.exit(1)

        self.get_logger().info("🤖 Math Verified. Executing physical sequence in 3 seconds...")
        time.sleep(3)
        
        for step in demo_sequence:
            self.move_to_cartesian_target(step["name"], step["x"], step["z"], step["gripper"])
            time.sleep(1.8) # Dwell delay to ensure stable trajectory settlement

        self.destroy_node()

    # --- SERIAL COMMUNICATIONS LAYER ---
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

    # --- MATH MAPPING TRANSLATION LAYER ---
    def degrees_to_ticks(self, degrees, motor_id):
        home_ticks = self.home_positions[motor_id]
        direction = self.ELBOW_DIRECTION if motor_id == 3 else self.SHOULDER_DIRECTION
        
        # 🔴 COMPENSATE FOR PHYSICAL ZERO OFFSET
        if motor_id == 2:
            degrees -= self.SHOULDER_HOME_ANGLE_DEG  
        
        target_ticks = int(round((degrees * direction * self.TICKS_PER_DEGREE) + home_ticks))
        return max(self.limits[motor_id][0], min(target_ticks, self.limits[motor_id][1]))

    # --- SPATIAL MATH ENGINE (ELBOW-UP INVERSE KINEMATICS) ---
    def calculate_ik(self, x, z):
        z_rel = z - self.base_height
        D = math.sqrt(x**2 + z_rel**2)
        
        if D > (self.L1 + self.L2) or D < abs(self.L1 - self.L2):
            return None, None

        cos_elbow_inner = (self.L1**2 + self.L2**2 - D**2) / (2 * self.L1 * self.L2)
        cos_elbow_inner = max(-1.0, min(1.0, cos_elbow_inner))
        gamma_elbow = math.acos(cos_elbow_inner)

        cos_shoulder_inner = (self.L1**2 + D**2 - self.L2**2) / (2 * self.L1 * D)
        cos_shoulder_inner = max(-1.0, min(1.0, cos_shoulder_inner))
        alpha_shoulder = math.acos(cos_shoulder_inner)

        beta_target = math.atan2(z_rel, x)

        theta_shoulder_deg = math.degrees(beta_target + alpha_shoulder)
        theta_elbow_deg = math.degrees(math.pi - gamma_elbow)

        return theta_shoulder_deg, theta_elbow_deg

    def move_to_cartesian_target(self, stage_name, x, z, gripper_ticks):
        self.get_logger().info(f"📍 Target Command: {stage_name} ➡️ X={x}cm, Z={z}cm")
        
        shoulder_deg, elbow_deg = self.calculate_ik(x, z)
        if shoulder_deg is None:
            return

        s_ticks = self.degrees_to_ticks(shoulder_deg, 2)
        e_ticks = self.degrees_to_ticks(elbow_deg, 3)

        self.write_position(2, s_ticks)
        self.write_position(3, e_ticks)
        self.write_position(6, gripper_ticks)

    def destroy_node(self):
        self.get_logger().info("🔓 Releasing joint torque parameters...")
        for i in [1, 2, 3, 6]:
            self.enable_torque(i, False)
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SO100InverseKinematicsNode()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()