"""Extract every video frame of the teleop dataset to frames/frame_XXXXXX.jpg for SO100Dataset."""
import glob
import os

import cv2

from so100_recorder import config

VIDEO_DIR = os.path.join(config.DATASET_DIR, 'videos', 'observation.images.main_camera', 'chunk-000')
OUT_DIR = os.path.join(config.DATASET_DIR, 'frames')


def main():
    vid_files = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Found {len(vid_files)} videos. Starting extraction...')

    # Frames are numbered globally across videos, matching the merged Parquet row index.
    global_idx = 0
    for vid in vid_files:
        print(f'Extracting {os.path.basename(vid)}...')
        cap = cv2.VideoCapture(vid)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(OUT_DIR, f'frame_{global_idx:06d}.jpg'), frame)
            global_idx += 1
            if global_idx % 2000 == 0:
                print(f'   -> Extracted {global_idx} frames so far...')
        cap.release()

    print(f'Done! Extracted {global_idx} images to {OUT_DIR}')


if __name__ == '__main__':
    main()
