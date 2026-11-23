#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from so100_recorder import config
from so100_recorder.Dataloader import SO100Dataset
from so100_recorder.diffusion_model import ConditionalUNet1D, DDPMScheduler, VisionEncoder

NUM_EPOCHS = 50
# Small batch to fit in laptop VRAM.
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CHECKPOINT_EVERY = 10


class DiffusionTrainerNode(Node):
    def __init__(self):
        super().__init__('diffusion_trainer_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Initializing diffusion training on: {self.device}')
        self.train_model()

    def save_weights(self, vision_encoder, unet, tag):
        torch.save(vision_encoder.state_dict(), os.path.join(config.MODEL_DIR, f'vision_encoder_{tag}.pth'))
        torch.save(unet.state_dict(), os.path.join(config.MODEL_DIR, f'unet_{tag}.pth'))

    def train_model(self):
        # --- Models ---
        vision_encoder = VisionEncoder(feature_dim=config.VISION_FEATURE_DIM).to(self.device)
        unet = ConditionalUNet1D(action_dim=config.ACTION_DIM,
                                 global_cond_dim=config.GLOBAL_COND_DIM).to(self.device)
        noise_scheduler = DDPMScheduler(num_timesteps=config.NUM_DIFFUSION_STEPS)

        # --- Data ---
        self.get_logger().info(f'Loading dataset from: {config.DATASET_DIR} ...')
        try:
            dataset = SO100Dataset(config.DATASET_DIR, chunk_size=config.CHUNK_SIZE)
            # num_workers=0 avoids fork/threading deadlocks with ROS 2.
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0, pin_memory=True)
        except Exception as e:
            self.get_logger().error(f'Failed to load dataset: {e}')
            return

        # --- Optimizer ---
        optimizer = torch.optim.AdamW(
            list(vision_encoder.parameters()) + list(unet.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        os.makedirs(config.MODEL_DIR, exist_ok=True)

        # --- Training loop ---
        self.get_logger().info('Starting training loop...')
        epoch_losses = []

        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            vision_encoder.train()
            unet.train()

            for batch in dataloader:
                img = batch['img'].to(self.device)
                state = batch['state'].to(self.device)
                # (B, seq, joints) -> (B, joints, seq) for Conv1d.
                actions = batch['actions'].to(self.device).transpose(1, 2)

                optimizer.zero_grad()

                # Condition on visual features + joint state.
                obs_features = torch.cat([vision_encoder(img), state], dim=-1)

                # Forward diffusion: noise the actions at random timesteps.
                noise = torch.randn_like(actions, device=self.device)
                timesteps = torch.randint(0, noise_scheduler.num_timesteps,
                                          (actions.shape[0],), device=self.device)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

                # Predict the noise and regress against the true noise.
                predicted_noise = unet(noisy_actions, timesteps, obs_features)
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            current_lr = scheduler.get_last_lr()[0]
            self.get_logger().info(
                f'Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}')

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                self.get_logger().info(f'Saving intermediate checkpoint for epoch {epoch + 1}...')
                self.save_weights(vision_encoder, unet, f'ep{epoch + 1}')

        # --- Final save and loss curve ---
        self.get_logger().info('Saving final model weights...')
        self.save_weights(vision_encoder, unet, 'final')

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, marker='o', markersize=4,
                 linestyle='-', color='#1f77b4', linewidth=2)
        plt.title('Diffusion Policy Training: Mean Squared Error (MSE) Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        graph_path = os.path.join(config.MODEL_DIR, 'training_loss_curve.png')
        plt.savefig(graph_path, dpi=300)
        plt.close()

        self.get_logger().info(f'Training complete! Loss curve saved to: {graph_path}')


def main(args=None):
    rclpy.init(args=args)
    node = DiffusionTrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Training manually interrupted.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
