#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import cv2
import numpy as np
import serial
import time
import os

# Import your trained model classes
from so100_recorder.diffusion_model import VisionEncoder, ConditionalUNet1D, DDPMScheduler

class DiffusionDeploymentNode(Node):
    def __init__(self):
        super().__init__('diffusion_deployment_node')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"🚀 Initializing Autonomous Deployment on: {self.device}")

        # ==========================================
        # 1. HARDWARE & LIMITS SETUP
        # ==========================================
        self.motor_ids = [1, 2, 3, 6]
        self.limits = {
            1: [0,    2500],
            2: [1698, 4090],
            3: [460,  2677],
            6: [3354, 4095]
        }
        
        # Normalization bounds for denormalization
        self.TICK_MIN = np.array([0., 1698., 460., 3354.], dtype=np.float32)
        self.TICK_MAX = np.array([2500., 4090., 2677., 4095.], dtype=np.float32)

        try:
            self.ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.5)
            self.get_logger().info("Serial Port Opened")
        except Exception as e:
            self.get_logger().error(f" Could not open serial port: {e}")
            exit(1)

        # Lock in the motors
        for i in self.motor_ids:
            self.enable_torque(i, True)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Force camera to drop old frames to prevent lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        if not self.cap.isOpened():
            self.get_logger().error("Could not open USB camera!")
            exit(1)

        
        
       
        self.vision_encoder = VisionEncoder(feature_dim=256).to(self.device)
        self.unet = ConditionalUNet1D(action_dim=4, global_cond_dim=260).to(self.device)
        self.noise_scheduler = DDPMScheduler(num_timesteps=100)

       
        EPOCH_TO_TEST = "final"  
        
        self.get_logger().info(f"Loading weights from Epoch {EPOCH_TO_TEST}...")
        if EPOCH_TO_TEST == "final":
            vis_weight = "vision_encoder_final.pth"
            unet_weight = "unet_final.pth"
        else:
            vis_weight = f"vision_encoder_ep{EPOCH_TO_TEST}.pth"
            unet_weight = f"unet_ep{EPOCH_TO_TEST}.pth"

        save_dir = os.path.expanduser("~/so100_ws/models")
        
    
        self.vision_encoder.load_state_dict(torch.load(
            os.path.join(save_dir, vis_weight), map_location=self.device, weights_only=True))
        self.unet.load_state_dict(torch.load(
            os.path.join(save_dir, unet_weight), map_location=self.device, weights_only=True))
        
        self.vision_encoder.eval()
        self.unet.eval()
        self.get_logger().info("Weights loaded successfully!")

        # ==========================================
        # 3. AUTONOMOUS EXECUTION PARAMETERS
        # ==========================================
        # Receding Horizon: Execute 4 steps per chunk
        self.T_exec = 4          
        # Execution Speed: ~20 Hz
        self.control_rate = 0.05 
        
        # Initialize the hardware smoothing filter
        self.current_smoothed_ticks = None
        
        self.get_logger().info(" SYSTEM READY. Starting main inference loop...")
        

    # --- Hardware Control Methods ---
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

    
    def get_current_joint_states(self):
        positions = []
        for motor_id in self.motor_ids:
            packet = [0xFF, 0xFF, motor_id, 4, 0x02, 0x38, 2]
            checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
            packet.append(checksum)
            self.ser.reset_input_buffer()
            self.ser.write(bytearray(packet))
            time.sleep(0.005)
            resp = self.ser.read(8)
            
            if len(resp) == 8 and resp[0] == 0xFF and resp[1] == 0xFF:
                pos = (resp[6] << 8) | resp[5]
                positions.append(float(pos))
            else:
                # Safe fallback if communication drops
                fallback = self.current_smoothed_ticks[len(positions)] if self.current_smoothed_ticks is not None else 2048.0
                positions.append(fallback)
        
        # Return as raw float32 tensor to exactly match Dataloader behavior
        return torch.tensor(np.array(positions, dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(self.device)

    # --- Neural Network Methods ---
    def get_live_image_tensor(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera frame dropped!")
        
        cv2.imshow("Diffusion Policy Autonomous View", frame)
        cv2.waitKey(1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        return img_tensor.to(self.device)

    def run_autonomous_loop(self):
        with torch.no_grad(): # No gradients needed for inference!
            while rclpy.ok():
                loop_start_time = time.time()

                # 1. OBSERVE
                img_tensor = self.get_live_image_tensor()
                visual_features = self.vision_encoder(img_tensor)
                
                state_tensor = self.get_current_joint_states()
                obs_features = torch.cat([visual_features, state_tensor], dim=-1)

                # 2. INITIALIZE NOISE (Batch 1, 4 Joints, 16 Steps)
                noisy_actions = torch.randn((1, 4, 16), device=self.device)

                # 3. DENOISE
                for t in reversed(range(self.noise_scheduler.num_timesteps)):
                    timesteps = torch.tensor([t], device=self.device)
                    predicted_noise = self.unet(noisy_actions, timesteps, obs_features)
                    
                    noisy_actions = self.noise_scheduler.step(predicted_noise, t, noisy_actions, inference=True)

                # 4. EXTRACT AND DENORMALIZE
                clean_action_chunk = noisy_actions.squeeze(0).transpose(0, 1).cpu().numpy()
                physical_actions = (clean_action_chunk + 1.0) / 2.0 * (self.TICK_MAX - self.TICK_MIN) + self.TICK_MIN

                # 5. EXECUTE WITH LOW-PASS HARDWARE FILTER
                for step_idx in range(4, 4 + self.T_exec): 
                    raw_target = physical_actions[step_idx]
                    
                    if self.current_smoothed_ticks is None:
                        self.current_smoothed_ticks = raw_target
                    else:
                        # EMA Filter: 40% trust in new prediction, 60% momentum from last position
                        self.current_smoothed_ticks = 0.4 * raw_target + 0.6 * self.current_smoothed_ticks

                    for j, motor_id in enumerate(self.motor_ids):
                        self.write_position(motor_id, self.current_smoothed_ticks[j])
                    
                    time.sleep(self.control_rate)

                latency = time.time() - loop_start_time
                self.get_logger().info(f"Policy cycle completed in {latency:.3f}s. Executed {self.T_exec} actions.")

    def destroy_node(self):
        self.get_logger().info(" Shutting down. Releasing motors...")
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
    node = DiffusionDeploymentNode()
    try:
       
        node.run_autonomous_loop()
    except KeyboardInterrupt:
        node.get_logger().info("Autonomous mode manually interrupted.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()