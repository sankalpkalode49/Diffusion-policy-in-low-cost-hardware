import cv2
import os
import glob

# Setup paths
vid_dir = os.path.expanduser("~/so100_dataset/so100_teleop/videos/observation.images.main_camera/chunk-000")
vid_files = glob.glob(os.path.join(vid_dir, "*.mp4"))
vid_files.sort()

out_dir = os.path.expanduser("~/so100_dataset/so100_teleop/frames")
os.makedirs(out_dir, exist_ok=True)

print(f"Found {len(vid_files)} videos. Starting extraction...")

global_idx = 0
for vid in vid_files:
    print(f"Extracting {os.path.basename(vid)}...")
    cap = cv2.VideoCapture(vid)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save as frame_000000.jpg, frame_000001.jpg, etc.
        img_path = os.path.join(out_dir, f"frame_{global_idx:06d}.jpg")
        cv2.imwrite(img_path, frame)
        global_idx += 1
        
        if global_idx % 2000 == 0:
            print(f"   -> Extracted {global_idx} frames so far...")
            
    cap.release()

print(f"✅ DONE! Successfully extracted {global_idx} images to {out_dir}")