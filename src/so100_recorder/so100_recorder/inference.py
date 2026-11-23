#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
import serial
import torch

from so100_recorder import config, feetech
from so100_recorder.diffusion_model import ConditionalUNet1D, DDPMScheduler, VisionEncoder

# Which checkpoint to load: 'final' or an epoch number such as 30.
CHECKPOINT = 'final'

# Receding horizon: execute T_EXEC actions from each predicted chunk,
# starting at ACTION_START_IDX.
ACTION_START_IDX = 4
T_EXEC = 4
CONTROL_PERIOD = 0.05  # seconds (~20 Hz)

# EMA weight given to each new prediction (the rest is momentum from the last command).
EMA_ALPHA = 0.4

FALLBACK_TICKS = 2048.0


class DiffusionDeploymentNode(Node):
    def __init__(self):
        super().__init__('diffusion_deployment_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Initializing autonomous deployment on: {self.device}')

        # --- Hardware ---
        self.motor_ids = config.ARM_MOTOR_IDS
        self.limits = config.TICK_LIMITS
        self.tick_min = np.array(config.TICK_MIN, dtype=np.float32)
        self.tick_max = np.array(config.TICK_MAX, dtype=np.float32)

        try:
            self.ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.5)
            self.get_logger().info('Serial port opened')
        except Exception as e:
            self.get_logger().error(f'Could not open serial port: {e}')
            raise SystemExit(1)

        for motor_id in self.motor_ids:
            self.enable_torque(motor_id, True)

        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        # Keep only the newest frame to avoid lag.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open USB camera!')
            raise SystemExit(1)

        # --- Model ---
        self.vision_encoder = VisionEncoder(feature_dim=config.VISION_FEATURE_DIM).to(self.device)
        self.unet = ConditionalUNet1D(action_dim=config.ACTION_DIM,
                                      global_cond_dim=config.GLOBAL_COND_DIM).to(self.device)
        self.noise_scheduler = DDPMScheduler(num_timesteps=config.NUM_DIFFUSION_STEPS)

        tag = 'final' if CHECKPOINT == 'final' else f'ep{CHECKPOINT}'
        self.get_logger().info(f'Loading weights from checkpoint {tag}...')
        self.vision_encoder.load_state_dict(torch.load(
            os.path.join(config.MODEL_DIR, f'vision_encoder_{tag}.pth'),
            map_location=self.device, weights_only=True))
        self.unet.load_state_dict(torch.load(
            os.path.join(config.MODEL_DIR, f'unet_{tag}.pth'),
            map_location=self.device, weights_only=True))
        self.vision_encoder.eval()
        self.unet.eval()
        self.get_logger().info('Weights loaded successfully!')

        # Last command sent to the servos, used by the EMA filter.
        self.current_smoothed_ticks = None

        self.get_logger().info('System ready. Starting main inference loop...')

    # --- Hardware ---
    def enable_torque(self, motor_id, enable=True):
        feetech.enable_torque(self.ser, motor_id, enable)
        time.sleep(0.01)

    def write_position(self, motor_id, position_ticks):
        lo, hi = self.limits[motor_id]
        feetech.write_position(self.ser, motor_id, max(lo, min(int(position_ticks), hi)))

    def get_current_joint_states(self):
        positions = []
        for j, motor_id in enumerate(self.motor_ids):
            pos = feetech.read_position(self.ser, motor_id, settle=0.005)
            if pos is not None:
                positions.append(float(pos))
            elif self.current_smoothed_ticks is not None:
                # Communication dropped: reuse the last command for this joint.
                positions.append(self.current_smoothed_ticks[j])
            else:
                positions.append(FALLBACK_TICKS)

        # Raw ticks as float32, matching what the Dataloader feeds the model.
        return torch.tensor(np.array(positions, dtype=np.float32)).unsqueeze(0).to(self.device)

    # --- Policy ---
    def get_live_image_tensor(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError('Camera frame dropped!')

        cv2.imshow('Diffusion Policy Autonomous View', frame)
        cv2.waitKey(1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        return img_tensor.to(self.device)

    def predict_action_chunk(self):
        """Run the full reverse diffusion and return a (CHUNK_SIZE, ACTION_DIM) array in ticks."""
        visual_features = self.vision_encoder(self.get_live_image_tensor())
        obs_features = torch.cat([visual_features, self.get_current_joint_states()], dim=-1)

        actions = torch.randn((1, config.ACTION_DIM, config.CHUNK_SIZE), device=self.device)
        for t in reversed(range(self.noise_scheduler.num_timesteps)):
            timesteps = torch.tensor([t], device=self.device)
            predicted_noise = self.unet(actions, timesteps, obs_features)
            actions = self.noise_scheduler.step(predicted_noise, t, actions, inference=True)

        chunk = actions.squeeze(0).transpose(0, 1).cpu().numpy()
        # Undo the [-1, 1] normalization.
        return (chunk + 1.0) / 2.0 * (self.tick_max - self.tick_min) + self.tick_min

    def run_autonomous_loop(self):
        with torch.no_grad():
            while rclpy.ok():
                loop_start_time = time.time()
                physical_actions = self.predict_action_chunk()

                for step_idx in range(ACTION_START_IDX, ACTION_START_IDX + T_EXEC):
                    raw_target = physical_actions[step_idx]
                    if self.current_smoothed_ticks is None:
                        self.current_smoothed_ticks = raw_target
                    else:
                        self.current_smoothed_ticks = (EMA_ALPHA * raw_target
                                                       + (1 - EMA_ALPHA) * self.current_smoothed_ticks)

                    for j, motor_id in enumerate(self.motor_ids):
                        self.write_position(motor_id, self.current_smoothed_ticks[j])

                    time.sleep(CONTROL_PERIOD)

                latency = time.time() - loop_start_time
                self.get_logger().info(f'Policy cycle completed in {latency:.3f}s. Executed {T_EXEC} actions.')

    def destroy_node(self):
        self.get_logger().info('Shutting down. Releasing motors...')
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
    node = DiffusionDeploymentNode()
    try:
        node.run_autonomous_loop()
    except KeyboardInterrupt:
        node.get_logger().info('Autonomous mode manually interrupted.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
