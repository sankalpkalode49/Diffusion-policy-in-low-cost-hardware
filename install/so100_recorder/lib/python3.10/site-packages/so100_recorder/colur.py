#!/usr/bin/env python3
import os
import cv2
from ultralytics import YOLO

def run_object_detection():
    # 1. Define input and output video paths
    input_path = os.path.expanduser("~/Downloads/presentation.mp4")
    output_path = os.path.expanduser("~/so100_ws/detected_objects_demo.mp4")

    if not os.path.exists(input_path):
        print(f"❌ Error: Cannot find dataset video at {input_path}")
        return

    # 2. Load the pre-trained YOLOv8 Nano model (Lightweight and fast)
    print("🧠 Loading pre-trained YOLOv8 Model weights...")
    model = YOLO("yolov8n.pt") 

    # 3. Open the video capture stream
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 100: fps = 30.0

    # 4. Initialize the video writer to save the results
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("🎬 Running object detection inference on video frames...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        # stream=True optimizes memory usage for video processing
        results = model(frame, verbose=False)

        # Plot the bounding boxes, labels, and confidence scores onto the frame
        annotated_frame = results[0].plot()

        # Write the processed frame to the output video
        out.write(annotated_frame)
        frame_count += 1

        # Optional: Display the live processing window (Press 'q' to quit early)
        cv2.imshow("YOLOv8 Real-Time Inference View", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Inference manually stopped by user.")
            break

    # 5. Release hardware and file structures
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"🎉 Success! Processed {frame_count} frames.")
    print(f"📂 Saved annotated object detection video to: {output_path}")

if __name__ == "__main__":
    run_object_detection()