import os
import shutil

# Base directory where frames are stored
BASE_DIR = "/home/lab-25/Desktop/Ml/Deep Learning/Deepfake/LivePortrait/animations_ffhq"

# Destination folder for selected frames
DEST_FOLDER = "/home/lab-25/Desktop/Ml/Deep Learning/Deepfake/LivePortrait/animations_ffhq/selected_frames"

# Create the destination folder if it doesn't exist
os.makedirs(DEST_FOLDER, exist_ok=True)

def move_selected_frames(base_dir, dest_folder):
    """Finds and moves only frames ending with _d0_frame1sec.jpg, skipping _d0_concat_frame1sec.jpg."""
    found_files = False  # Track if any files are found

    for root, _, files in os.walk(base_dir):  # Walk through all subfolders
        for file in files:
            if file.endswith("--d0_frame1sec.jpg") and not file.endswith("_d0_concat_frame1sec.jpg"):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)

                # Move the file to the destination folder
                shutil.move(source_path, dest_path)
                print(f"✅ Moved: {source_path} → {dest_path}")
                found_files = True

    if not found_files:
        print("⚠️ No matching frames found!")

if __name__ == "__main__":
    move_selected_frames(BASE_DIR, DEST_FOLDER)
    print("🎉 Frame selection and moving completed!")

