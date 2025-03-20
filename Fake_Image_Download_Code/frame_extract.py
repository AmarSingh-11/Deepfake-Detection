import os
import cv2

# Define the base directory where your videos are stored
BASE_DIR = "/home/lab-25/Desktop/Ml/Deep Learning/Deepfake/LivePortrait/animations_ffhq"

def extract_frame(video_path, output_image_path, time_sec=1):
    """Extracts a frame at a specific time from a video and saves it as an image."""
    cap = cv2.VideoCapture(video_path)

    # Check if video file is opened successfully
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return

    # Set the capture position to the desired time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

    # Read the frame
    success, frame = cap.read()

    if success:
        cv2.imwrite(output_image_path, frame)  # Save the frame
        print(f"✅ Extracted frame from: {video_path} → Saved as: {output_image_path}")
    else:
        print(f"❌ Error: Could not extract frame from {video_path}")

    # Release the video capture object
    cap.release()

def process_videos(base_dir):
    """Finds all .mp4 files in the directory (excluding *_d0_concat.mp4) and extracts a frame."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4") and "_d0_concat.mp4" not in file:
                video_path = os.path.join(root, file)
                output_image_path = os.path.join(root, f"{os.path.splitext(file)[0]}_frame1sec.jpg")

                extract_frame(video_path, output_image_path)

if __name__ == "__main__":
    process_videos(BASE_DIR)
    print("🎉 Frame extraction completed!")

