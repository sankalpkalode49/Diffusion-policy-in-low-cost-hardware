import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

from so100_recorder.Dataloader import SO100Dataset
from so100_recorder.diffusion_model import VisionEncoder, ConditionalUNet1D, DDPMScheduler

class DiffusionTrainerNode(Node):
    def __init__(self):
        super().__init__('diffusion_trainer_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"🚀 Initializing Training on: {self.device}")
        self.train_model()

    def train_model(self):
        vision_encoder = VisionEncoder().to(self.device)
        unet = ConditionalUNet1D().to(self.device)
        noise_scheduler = DDPMScheduler(num_timesteps=100)

        dataset_path = os.path.expanduser("~/so100_dataset/so100_teleop")
        
        try:
            dataset = SO100Dataset(dataset_path, chunk_size=16)
            # Physical batch size remains 2 to protect laptop VRAM
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")
            return

        optimizer = torch.optim.AdamW(
            list(vision_encoder.parameters()) + list(unet.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        num_epochs = 50
        epoch_losses = []
        
        # 🟡 NEW: Gradient Accumulation (8 steps * batch_size 2 = Effective Batch 16)
        accumulation_steps = 8 
        
        save_dir = os.path.expanduser("~/so100_ws/models")
        os.makedirs(save_dir, exist_ok=True)
        
        self.get_logger().info("🔥 Starting Training Loop...")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            optimizer.zero_grad() 

            for batch_idx, batch in enumerate(dataloader):
                img = batch['img'].to(self.device)
                actions = batch['actions'].to(self.device) 
                actions = actions.transpose(1, 2) 

                obs_features = vision_encoder(img)
                
                # 🔴 FIXED: Removed device=self.device to prevent TypeError
                noise = torch.randn_like(actions)
                
                bsz = actions.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (bsz,), device=self.device)

                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                predicted_noise = unet(noisy_actions, timesteps, obs_features)

                # --- BACKWARD PASS ---
                loss = F.mse_loss(predicted_noise, noise)
                loss = loss / accumulation_steps # Scale loss for accumulation
                loss.backward()

                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                    # 🟡 NEW: Gradient Clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(
                        list(vision_encoder.parameters()) + list(unet.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += (loss.item() * accumulation_steps)

            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            self.get_logger().info(f"Epoch [{epoch+1}/{num_epochs}] | Average Loss: {avg_loss:.5f}")
            
            # 🟡 NEW: Checkpointing every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.get_logger().info(f"💾 Saving checkpoint at epoch {epoch + 1}...")
                torch.save(vision_encoder.state_dict(), os.path.join(save_dir, f"vision_encoder_ep{epoch+1}.pth"))
                torch.save(unet.state_dict(), os.path.join(save_dir, f"unet_ep{epoch+1}.pth"))

        self.get_logger().info("💾 Saving final model weights...")
        torch.save(vision_encoder.state_dict(), os.path.join(save_dir, "vision_encoder_final.pth"))
        torch.save(unet.state_dict(), os.path.join(save_dir, "unet_final.pth"))
        
        # --- GENERATE THE REPORT GRAPH ---
        self.get_logger().info("📊 Generating training loss graph...")
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
        
        self.get_logger().info(f"✅ Training complete! Graph saved to: {graph_path}")

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