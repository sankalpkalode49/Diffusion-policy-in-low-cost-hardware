# SO-100 Diffusion Policy for Robotic Manipulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-green)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete end-to-end implementation of **Diffusion Policy** for real-world robotic manipulation, built from scratch in PyTorch and deployed on a custom 4-DOF SO-100 robotic arm via ROS2. This project demonstrates visuomotor policy learning: the robot observes through a camera, reasons about its joint state, and generates smooth, temporally coherent action trajectories for autonomous pick-and-place tasks --- without explicit inverse kinematics or trajectory optimization.

https://github.com/user-attachments/assets/presentation.mp4

---

## What is Diffusion Policy?

Diffusion Policy treats robot action generation as a **denoising diffusion process** in the action space. Instead of regressing a single action, it learns to iteratively denoise random action sequences conditioned on visual and proprioceptive observations. This captures multi-modal behavior distributions (e.g., multiple valid grasp approaches) and produces smooth, human-like motion through action chunking.

Key advantages over traditional BC/IL:
- **Multi-modal action distributions** via diffusion in action space
- **Temporal coherence** through action chunking (predicting future action sequences)
- **Smooth execution** without explicit trajectory optimization

---

## System Architecture

```
+-----------------+     +-----------------+     +-----------------+
|  Data Collection |---->|     Training     |---->|    Deployment   |
|   (teleop_rec)   |     |   (train.py)     |     |  (inference.py) |
+-----------------+     +-----------------+     +-----------------+
         |                       |                       |
         v                       v                       v
   PS5 Controller          DDPM + U-Net           Real-time ROS2
   LeRobot Dataset         ResNet18 Encoder       Servo Control
   Multi-camera sync        Action Chunking        EMA Smoothing
```

### Hardware Stack
| Component | Specification |
|-----------|---------------|
| Robot Arm | SO-100 (4-DOF: base pan, shoulder lift, elbow flex, gripper) |
| Actuators | FeeTech STS3215 servos (position control, 0-4095 ticks) |
| Camera | USB webcam (640x480 @ 30 FPS) |
| Compute | NVIDIA Jetson / Laptop with CUDA |
| Interface | USB-to-TTL serial (`/dev/ttyACM0`) |
| Control | ROS2 Humble |

### Software Stack
| Module | Technology |
|--------|------------|
| Deep Learning | PyTorch, TorchVision |
| Diffusion | Custom DDPM scheduler (100 timesteps) |
| Vision | ResNet18 backbone + linear projection |
| Policy Network | Conditional 1D U-Net with FiLM conditioning |
| Robotics | ROS2 Humble, `rclpy`, `sensor_msgs`, `cv_bridge` |
| Dataset | LeRobot format (Parquet + video chunks) |
| Deployment | Real-time inference at ~20Hz |

---

## Repository Structure

```
so100-diffusion-policy/
├── src/so100_recorder/
│   ├── diffusion_model.py      # Core diffusion model (Vision Encoder, DDPM, U-Net)
│   ├── Dataloader.py           # SO100Dataset: Parquet loading, action normalization
│   ├── train.py                # ROS2-integrated training loop with checkpointing
│   ├── inference.py            # Real-time autonomous deployment node
│   ├── teleop_rec.py           # PS5 teleoperation + LeRobot dataset recording
│   ├── episode_rec.py          # Multi-camera episode recorder (overhead + wrist)
│   ├── scan.py                 # Hardware diagnostic: servo scan & wiggle test
│   └── extract_frame.py        # Video-to-frame extraction utility
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

---

## Key Features

### 1. From-Scratch Diffusion Policy
- **Custom DDPM scheduler** with forward noise injection and reverse denoising
- **Conditional 1D U-Net** with sinusoidal timestep embeddings and FiLM conditioning
- **ResNet18 vision encoder** pretrained on ImageNet, fine-tuned for robot manipulation
- **Action chunking**: Predicts 16-step action sequences for smooth execution

### 2. Action Space Normalization
Servo commands (ticks) are normalized to `[-1, 1]` for stable diffusion training:
```python
TICK_MIN = [0, 1698, 460, 3354]
TICK_MAX = [2500, 4090, 2677, 4095]
actions = 2.0 * (actions - TICK_MIN) / (TICK_MAX - TICK_MIN) - 1.0
```

### 3. Real-Time Deployment
- **Receding horizon control**: Executes 4 steps per generated chunk, then re-plans
- **EMA hardware smoothing**: 40% new prediction / 60% momentum for jitter-free motion
- **Serial fallback**: Safe position hold on communication dropout
- **Deterministic inference**: Zero noise injection during deployment (`inference=True`)

### 4. ROS2 Integration
- Full `rclpy` node lifecycle with proper cleanup
- QoS-optimized camera subscriptions (`qos_profile_sensor_data`)
- PS5 controller teleoperation with debounced recording toggle
- Multi-camera support (overhead + wrist) for rich observation space

---

## Setup & Installation

### Prerequisites
- Ubuntu 22.04 with ROS2 Humble
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- SO-100 robot arm + USB camera + PS5 controller

### 1. Clone & Install
```bash
git clone https://github.com/sankalpkalode49/Diffusion-policy-in-low-cost-hardware.git
cd Diffusion-policy-in-low-cost-hardware
pip install -r requirements.txt
```

### 2. ROS2 Workspace Setup
```bash
# Create workspace
mkdir -p ~/so100_ws/src
cd ~/so100_ws/src
ln -s /path/to/Diffusion-policy-in-low-cost-hardware .

# Build
cd ~/so100_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Hardware Permissions
```bash
# Add user to dialout group for serial access
sudo usermod -a -G dialout $USER
# Log out and back in for changes to take effect
```

---

## Usage

### 1. Hardware Test
Verify servo communication before training:
```bash
python3 src/so100_recorder/scan.py
```
This scans all 6 servo IDs and runs a safe wiggle sequence.

### 2. Data Collection
Teleoperate the arm and record demonstrations:
```bash
# Terminal 1: Start ROS2
cd ~/so100_ws && source install/setup.bash
ros2 run so100_recorder teleop_rec

# Terminal 2: Launch camera (if using ROS2 camera node)
ros2 run v4l2_camera v4l2_camera_node
```
**Controls:**
- `D-Pad` --- Shoulder pan / lift
- `△ / ✕` --- Elbow flex up / down
- `L2 / R2` --- Gripper open / close
- `R1` --- Start / stop recording
- `□` --- Return to home position

### 3. Frame Extraction
Convert recorded video chunks to frame images:
```bash
python3 src/so100_recorder/extract_frame.py
```

### 4. Training
Launch the diffusion policy training node:
```bash
cd ~/so100_ws && source install/setup.bash
ros2 run so100_recorder train
```

**Training Configuration:**
| Hyperparameter | Value |
|----------------|-------|
| Epochs | 50 |
| Batch Size | 2 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| LR Scheduler | Cosine Annealing |
| Chunk Size | 16 |
| Diffusion Timesteps | 100 |
| Vision Feature Dim | 256 |

Checkpoints save every 10 epochs to `~/so100_ws/models/`.

### 5. Autonomous Deployment
Run the trained policy on hardware:
```bash
cd ~/so100_ws && source install/setup.bash
ros2 run so100_recorder inference
```

The robot will:
1. Capture live camera frame
2. Extract visual features + current joint state
3. Generate 16-step action chunk via DDPM denoising
4. Execute first 4 steps with EMA smoothing
5. Repeat

---

## Results

| Metric | Value |
|--------|-------|
| Task | Pick-and-place (box → cylinder) |
| Control Frequency | ~20 Hz |
| Inference Device | NVIDIA Jetson / CUDA GPU |
| Action Chunk Size | 16 steps |
| Execution Horizon | 4 steps per replan |
| Training Data | Human teleoperation demonstrations |
| Success Criteria | Grasp + transport + release without explicit IK |

---

## Implementation Details

### Vision Encoder
```python
class VisionEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.compress = nn.Linear(512, feature_dim)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
```
- Pretrained ResNet18 backbone (ImageNet)
- Global average pooling → 512-dim → compressed to 256-dim
- Standard ImageNet normalization for transfer learning stability

### Conditional 1D U-Net
```python
class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim=4, global_cond_dim=260):
        # 260 = 256 (vision) + 4 (proprioception)
        # Encoder-decoder with skip connections
        # FiLM conditioning via time + observation embeddings
```
- Sinusoidal positional embeddings for diffusion timesteps
- Observation conditioning through FiLM (Feature-wise Linear Modulation)
- Skip connections preserve action sequence structure

### DDPM Scheduler
```python
class DDPMScheduler:
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=2e-2):
        # Linear beta schedule
        # Forward: q(x_t | x_0) via closed-form
        # Reverse: p(x_{t-1} | x_t) with learned noise prediction
```

---

## Safety Notes

⚠️ **This project involves real hardware. Always:**
- Run `scan.py` first to verify servo communication
- Keep the emergency power switch accessible
- Start with small `speed_multiplier` values
- Use the EMA filter to prevent jerky motion
- Test in simulation/Gazebo before hardware deployment

---

## Future Work

- [ ] Integrate SLAM (`slam_toolbox` / `nav2`) for mobile base navigation
- [ ] Add depth camera (Intel RealSense) for 3D spatial reasoning
- [ ] Implement classifier-free guidance for behavior conditioning
- [ ] Train on multi-task dataset with task embeddings
- [ ] Add collision detection and hardware emergency stop
- [ ] Optimize to <50ms inference via TensorRT + ONNX

---

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) --- Dataset format and tooling
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) --- Original paper by Chi et al.
- [SO-100 Arm](https://github.com/TheRobotStudio/SO-ARM100) --- Open-source robot design

---

## License

MIT License --- See [LICENSE](LICENSE) for details.

---

## Contact

**Sankalp Kalode**  
📧 kalodesankalp909@gmail.com  
🔗 [linkedin.com/in/sankalp-kalode](https://linkedin.com/in/sankalp-kalode)  
💻 [github.com/sankalpkalode49](https://github.com/sankalpkalode49)
