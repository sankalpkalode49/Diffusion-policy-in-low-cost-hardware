"""Shared hardware, dataset and model settings for the SO-100 arm.

Training (Dataloader.py) and deployment (inference.py) must use the same
tick bounds and model dimensions, so they are defined once here.
"""
import os

# --- Serial bus ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 1000000

# --- 4-axis arm: base pan, shoulder lift, elbow flex, gripper ---
ARM_MOTOR_IDS = [1, 2, 3, 6]

# Safe position range per servo, in ticks (0-4095).
TICK_LIMITS = {
    1: (0, 2500),     # Base pan
    2: (1698, 4090),  # Shoulder lift (transplanted servo)
    3: (460, 2677),   # Elbow flex
    6: (3354, 4095),  # Gripper
}

HOME_POSITIONS = {1: 1250, 2: 2894, 3: 1568, 6: 3724}

# Per-joint bounds used to normalize actions to [-1, 1], ordered like ARM_MOTOR_IDS.
TICK_MIN = [float(TICK_LIMITS[m][0]) for m in ARM_MOTOR_IDS]
TICK_MAX = [float(TICK_LIMITS[m][1]) for m in ARM_MOTOR_IDS]

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Dataset ---
DATASET_ROOT = os.path.expanduser('~/so100_dataset')
DATASET_NAME = 'so100_teleop'
DATASET_DIR = os.path.join(DATASET_ROOT, DATASET_NAME)
DATASET_FPS = 30


def next_free_dataset_path(base_name=DATASET_NAME):
    """Return (repo_id, path) for the first DATASET_ROOT/<base_name>[_N] that does not exist yet."""
    repo_id = base_name
    path = os.path.join(DATASET_ROOT, repo_id)
    counter = 1
    while os.path.exists(path):
        repo_id = f'{base_name}_{counter}'
        path = os.path.join(DATASET_ROOT, repo_id)
        counter += 1
    return repo_id, path


# --- Model ---
ACTION_DIM = len(ARM_MOTOR_IDS)
CHUNK_SIZE = 16
VISION_FEATURE_DIM = 256
GLOBAL_COND_DIM = VISION_FEATURE_DIM + ACTION_DIM
NUM_DIFFUSION_STEPS = 100
MODEL_DIR = os.path.expanduser('~/so100_ws/models')
