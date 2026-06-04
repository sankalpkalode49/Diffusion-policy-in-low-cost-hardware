#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import serial
import time

class UnifiedLeRobotStyleTeleop(Node):
    def __init__(self):
        super().__init__('teleop_ps5')
        
        self.get_logger().info("🚀 Starting Unified PS5 Teleop Node (Hardware Limits Only)...")

        try:
            # Extended timeout slightly to allow for initial reading
            self.ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.5)
            self.get_logger().info("✅ Serial Port Opened")
        except Exception as e:
            self.get_logger().error(f"❌ Could not open serial port: {e}")
            return

        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # 🏎️ Speed: Ticks to move per loop. Lowered for smoothness.
        self.speed_multiplier = 8.0 

        # 🎮 Hardware State Tracking (Fixes Linux joy_node 0.0 boot bug for triggers)
        self.triggers_initialized = {2: False, 5: False}

        # 🔍 AUTO-READ STARTUP: Asks the motors exactly where they are
        self.targets = self.read_initial_positions()

        self.timer = self.create_timer(0.02, self.control_loop)

    def read_initial_positions(self):
        """ Asks each motor where it currently is so we don't snap on startup """
        self.get_logger().info("🔍 Reading physical resting position of the arm...")
        targets = {}
        for motor_id in range(1, 7):
            self.ser.reset_input_buffer()
            # Feetech Read Packet: Address 56 (0x38) for Present Position
            packet = [0xFF, 0xFF, motor_id, 0x04, 0x02, 0x38, 0x02]
            checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
            packet.append(checksum)
            
            self.ser.write(bytearray(packet))
            response = self.ser.read(8)
            
            if len(response) == 8 and response[0] == 0xFF and response[1] == 0xFF:
                pos = response[5] | (response[6] << 8)
                targets[motor_id] = float(pos)
                self.get_logger().info(f"📍 Motor {motor_id} found resting at tick {pos}")
            else:
                # Absolute fallback (assuming center of 0-4095 if disconnected)
                targets[motor_id] = 2048.0
                self.get_logger().warning(f"⚠️ Motor {motor_id} read failed. Defaulting to 2048")
            time.sleep(0.01)
            
        return targets

    def apply_deadzone(self, value, threshold=0.08):
        """ Acts as a digital brake: returns 0.0 if the stick is just drifting """
        return value if abs(value) > threshold else 0.0

    def joy_callback(self, msg):
        """ Maps PS5 joystick inputs directly to raw tick increments """
        
        # 1. Apply deadzone to filter out all hardware drift for the sticks
        ax = [self.apply_deadzone(a) for a in msg.axes]
        
        # 2. Add to targets based on clean joystick position
        self.targets[1] += ax[0] * self.speed_multiplier
        self.targets[2] += ax[1] * self.speed_multiplier
        self.targets[3] += ax[4] * self.speed_multiplier
        self.targets[4] += ax[3] * self.speed_multiplier
        self.targets[5] += ax[7] * self.speed_multiplier

        # 3. Trigger Initialization Fix (The Linux joy_node boot quirk)
        # Mark trigger as initialized on the first real hardware event
        for idx in [2, 5]:
            if msg.axes[idx] != 0.0:
                self.triggers_initialized[idx] = True
                
        # Default to 1.0 (fully released) if the hardware hasn't woken up yet
        l2_raw = msg.axes[2] if self.triggers_initialized[2] else 1.0
        r2_raw = msg.axes[5] if self.triggers_initialized[5] else 1.0        
        
        l2_val = (1.0 - l2_raw) / 2.0
        r2_val = (1.0 - r2_raw) / 2.0
        self.targets[6] += (l2_val - r2_val) * (self.speed_multiplier * 0.5)

        # NOTE: Software limits have been completely removed.
        # The script relies entirely on the STS3215 internal EEPROM limits.

    def write_position(self, motor_id, position_ticks):
        """ Sends direct serial packet with INTERPOLATION TIME to eliminate jitter """
        # Prevent Python from sending negative ticks or exceeding 4095
        # (This is a protocol requirement, not a physical workspace limit)
        safe_ticks = max(0, min(int(position_ticks), 4095))
        
        pos_low = safe_ticks & 0xFF
        pos_high = (safe_ticks >> 8) & 0xFF
        
        # 20ms interpolation tells the motor to glide to the next point
        time_ms = 20 
        time_low = time_ms & 0xFF
        time_high = (time_ms >> 8) & 0xFF
        
        speed_low = 0
        speed_high = 0
        
        packet = [0xFF, 0xFF, motor_id, 0x09, 0x03, 0x2A, 
                  pos_low, pos_high, 
                  time_low, time_high, 
                  speed_low, speed_high]
        
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        
        self.ser.write(bytearray(packet))

    def control_loop(self):
        """ Sends targets 50 times a second """
        for motor_id, target_ticks in self.targets.items():
            self.write_position(motor_id, target_ticks)
            time.sleep(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedLeRobotStyleTeleop()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down teleop...")
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()