#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# Import your dataset and model structures
from so100_recorder.Dataloader import SO100Dataset
from so100_recorder.diffusion_model import VisionEncoder, ConditionalUNet1D, DDPMScheduler

class DiffusionTrainerNode(Node):
    def __init__(self):
        super().__init__('diffusion_trainer_node')
        
        # 1. Setup Compute Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"🚀 Initializing Diffusion Training on: {self.device}")
        
        # Start the training sequence immediately
        self.train_model()

    def train_model(self):
        # ==========================================
        # 2. INITIALIZE MODELS
        # ==========================================
        # Vision outputs 256 dims. State provides 4 dims. Total global_cond = 260.
        vision_encoder = VisionEncoder(feature_dim=256).to(self.device)
        unet = ConditionalUNet1D(action_dim=4, global_cond_dim=260).to(self.device)
        noise_scheduler = DDPMScheduler(num_timesteps=100)

        # ==========================================
        # 3. SETUP DATASET & DATALOADER
        # ==========================================
        dataset_path = os.path.expanduser("~/so100_dataset/so100_teleop")
        self.get_logger().info(f"Loading dataset from: {dataset_path} ...")
        
        try:
            dataset = SO100Dataset(dataset_path, chunk_size=16)
            # Physical batch size is 2 to protect laptop VRAM. 
            # num_workers=0 prevents ROS2 fork/threading deadlocks on local environments.
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")
            return

        # ==========================================
        # 4. OPTIMIZER & SCHEDULER
        # ==========================================
        num_epochs = 50
        
        optimizer = torch.optim.AdamW(
            list(vision_encoder.parameters()) + list(unet.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # Cosine Annealing smoothly decays the learning rate to prevent getting stuck in local minima
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # ==========================================
        # 5. THE MAIN TRAINING LOOP
        # ==========================================
        self.get_logger().info("🔥 Starting Training Loop...")
        epoch_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Set models to training mode
            vision_encoder.train()
            unet.train()

            for batch_idx, batch in enumerate(dataloader):
                # Move batches to GPU
                img = batch['img'].to(self.device)
                state = batch['state'].to(self.device)   # Proprioceptive joint states
                actions = batch['actions'].to(self.device)
                
                # Transpose actions for Conv1d: (Batch, sequence, joints) -> (Batch, joints, sequence)
                actions = actions.transpose(1, 2)
                
                optimizer.zero_grad()

                # --- FORWARD PASS & OBSERVATION FUSION ---
                visual_features = vision_encoder(img)
                
                # Concatenate 256-dim visual features with 4-dim joint state
                obs_features = torch.cat([visual_features, state], dim=-1)

                # --- NOISE INJECTION (DDPM FORWARD PROCESS) ---
                noise = torch.randn_like(actions, device=self.device)
                bsz = actions.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (bsz,), device=self.device)
                
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                # --- NOISE PREDICTION ---
                predicted_noise = unet(noisy_actions, timesteps, obs_features)

                # --- BACKWARD PASS ---
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            # Step the learning rate scheduler at the end of each epoch
            scheduler.step()

            # --- LOGGING & CHECKPOINTING ---
            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            current_lr = scheduler.get_last_lr()[0]
            
            self.get_logger().info(f"Epoch [{epoch+1:02d}/{num_epochs}] | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")

            # Save intermediate checkpoints every 10 epochs
            save_dir = os.path.expanduser("~/so100_ws/models")
            os.makedirs(save_dir, exist_ok=True)
            
            if (epoch + 1) % 10 == 0:
                self.get_logger().info(f" Saving intermediate checkpoint for Epoch {epoch+1}...")
                torch.save(vision_encoder.state_dict(), os.path.join(save_dir, f"vision_encoder_ep{epoch+1}.pth"))
                torch.save(unet.state_dict(), os.path.join(save_dir, f"unet_ep{epoch+1}.pth"))

        # ==========================================
        # 6. FINAL SAVE & GRAPH GENERATION
        # ==========================================
        self.get_logger().info(" Saving final model weights...")
        torch.save(vision_encoder.state_dict(), os.path.join(save_dir, "vision_encoder_final.pth"))
        torch.save(unet.state_dict(), os.path.join(save_dir, "unet_final.pth"))
        
        self.get_logger().info(" Generating training loss graph for the final report...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', markersize=4, linestyle='-', color='#1f77b4', linewidth=2)
        plt.title('Diffusion Policy Training: Mean Squared Error (MSE) Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        graph_path = os.path.join(save_dir, "training_loss_curve.png")
        plt.savefig(graph_path, dpi=300)
        plt.close()
        
        self.get_logger().info(f" Training complete! Graph saved to: {graph_path}")


def main(args=None):
    rclpy.init(args=args)
    node = DiffusionTrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Training manually interrupted.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()