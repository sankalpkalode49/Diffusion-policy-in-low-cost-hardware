#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import time
import os
import cv2  # <-- 1. Import OpenCV

class TrajectoryPlayer(Node):
    def __init__(self):
        super().__init__('trajectory_player')
        self.get_logger().info("🚀 Trajectory Playback Node Initialized.")

        try:
            # Fixed the port string just in case it was missing the '0'
            self.ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.5)
        except Exception as e:
            self.get_logger().error(f"Serial error: {e}")
            exit(1)

        # --- 2. SETUP DIRECT CAMERA (OpenCV) ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().warn("⚠️ Could not open camera! Playback will continue blindly.")

        # Structure-holding torque on the dead base motor (Motor 1)
        self.enable_torque(1, True)
        for i in [2, 3, 6]:
            self.enable_torque(i, True)

        file_path = os.path.expanduser('~/so100_ws/trajectory.txt')
        if not os.path.exists(file_path):
            self.get_logger().error(f"❌ No trajectory file found at {file_path}! Record one first.")
            exit(1)

        with open(file_path, 'r') as f:
            self.lines = f.readlines()

        self.get_logger().info(f"📋 Loaded {len(self.lines)} frames. Starting playback in 3 seconds...")
        time.sleep(3)

        self.run_playback()

    def enable_torque(self, motor_id, enable=True):
        val = 1 if enable else 0
        packet = [0xFF, 0xFF, motor_id, 0x04, 0x03, 0x28, val]
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        self.ser.write(bytearray(packet))
        time.sleep(0.01)

    def write_position(self, motor_id, position_ticks):
        pos_low  = position_ticks & 0xFF
        pos_high = (position_ticks >> 8) & 0xFF
        packet = [0xFF, 0xFF, motor_id, 0x05, 0x03, 0x2A, pos_low, pos_high]
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        self.ser.write(bytearray(packet))

    def run_playback(self):
        # self.get_logger().info("🎬 PLAYING BACK MOVEMENT AND CAMERA NOW...")
        
        target_hz = 30.0
        target_duration = 1.0 / target_hz

        for line in self.lines:
            loop_start = time.time() # Start timing this frame

            try:
                # 1. Parse and send motor commands
                s_ticks, e_ticks, g_ticks = map(int, line.strip().split(','))
                self.write_position(2, s_ticks)
                self.write_position(3, e_ticks)
                self.write_position(6, g_ticks)
                
                # 2. Read and display camera frame
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        cv2.imshow("Live Presentation View", frame)
                        # waitKey(1) is required to actually render the frame to the screen
                        cv2.waitKey(1) 

                # 3. Dynamic Sleep: Only sleep for the remaining time in our 33ms window
                elapsed = time.time() - loop_start
                sleep_time = target_duration - elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.get_logger().error(f"Playback error parsing frame: {e}")
                
        # self.get_logger().info("🎉 Playback complete! Operation successful.")
        self.destroy_node()

    def destroy_node(self):
        self.get_logger().info("Cleaning up hardware locks and closing windows...")
        for i in [1, 2, 3, 6]: 
            self.enable_torque(i, False)
        
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            
        # 4. Release camera hardware
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlayer()
    if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()