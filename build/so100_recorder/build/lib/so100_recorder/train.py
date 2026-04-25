import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

# Import your custom modules
# (You may need to prefix these with your package name, e.g., 'from so100_recorder.dataloader import ...')
from Dataloader import SO100Dataset
from diffusion_model import VisionEncoder, ConditionalUNet1D, DDPMScheduler


class DiffusionTrainerNode(Node):
    def __init__(self):
        super().__init__('diffusion_trainer_node')
        
        # 1. Setup Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"🚀 Initializing Training on: {self.device}")

        # Execute the training sequence
        self.train_model()

    def train_model(self):
        # 2. Initialize Models
        vision_encoder = VisionEncoder().to(self.device)
        unet = ConditionalUNet1D().to(self.device)
        noise_scheduler = DDPMScheduler(num_timesteps=100)

        # 3. Setup Dataset and DataLoader
        dataset_path = os.path.expanduser("~/so100_dataset/so100_teleop")
        self.get_logger().info(f"Loading dataset from: {dataset_path} ...")
        
        try:
            dataset = SO100Dataset(dataset_path, chunk_size=16)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")
            return

        # 4. Setup Optimizer
        optimizer = torch.optim.AdamW(
            list(vision_encoder.parameters()) + list(unet.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        # 5. The Training Loop
        num_epochs = 50
        self.get_logger().info("🔥 Starting Training Loop...")

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(dataloader):
                # Move data to GPU
                overhead_img = batch['overhead_img'].to(self.device)
                wrist_img = batch['wrist_img'].to(self.device)
                actions = batch['actions'].to(self.device) 

                # Transpose for Conv1d: (Batch, sequence, joints) -> (Batch, joints, sequence)
                actions = actions.transpose(1, 2) 

                optimizer.zero_grad()

                # --- FORWARD PASS ---
                obs_features = vision_encoder(overhead_img, wrist_img)
                noise = torch.randn_like(actions, device=self.device)
                
                bsz = actions.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (bsz,), device=self.device)

                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                predicted_noise = unet(noisy_actions, timesteps, obs_features)

                # --- BACKWARD PASS ---
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Log progress
            avg_loss = epoch_loss / len(dataloader)
            self.get_logger().info(f"Epoch [{epoch+1}/{num_epochs}] | Average Loss: {avg_loss:.5f}")

        # 6. Save the trained weights
        self.get_logger().info("💾 Saving model weights...")
        
        save_dir = os.path.expanduser("~/so100_ws/models")
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(vision_encoder.state_dict(), os.path.join(save_dir, "vision_encoder.pth"))
        torch.save(unet.state_dict(), os.path.join(save_dir, "unet.pth"))
        
        self.get_logger().info("✅ Training complete! Weights safely stored on disk.")


def main(args=None):
    rclpy.init(args=args)
    
    node = DiffusionTrainerNode()
    
    try:
        # Since training blocks the thread entirely, we don't actually need rclpy.spin(node) here, 
        # because train_model() runs completely inside the __init__ function before spin() is reached.
        # But we keep it for standard ROS 2 architecture in case you add threaded callbacks later.
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Training manually interrupted.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()