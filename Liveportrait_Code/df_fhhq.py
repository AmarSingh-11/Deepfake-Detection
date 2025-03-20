import os
from glob import glob
import subprocess

# Path to FFHQ images
ffhq_path = '/home/lab-25/Videos/fac'

# Path to driving video
driving_video = 'assets/examples/driving/d0.mp4'

# Output directory
output_dir = 'animations_fac/'
os.makedirs(output_dir, exist_ok=True)

# Get list of all images
images = glob(os.path.join(ffhq_path, '*.jpg'))

for img_path in images:
    img_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f'{img_name}_animation.mp4')
    
    # Skip if output file already exists
    if os.path.exists(output_path):
        print(f"Skipping {img_name}, already processed.")
        continue
    
    print(f"Processing {img_name}...")
    
    command = [
        'python', 'inference.py',
        '-s', img_path,
        '-d', driving_video,
        '-o', output_path
    ]
    subprocess.run(command)

