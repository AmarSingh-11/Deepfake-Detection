import os
import requests

# Corrected directory path
save_dir = ""
os.makedirs(save_dir, exist_ok=True)

# URL of the website
url = "https://thispersondoesnotexist.com/"

# Number of images to download
num_images = 1000

for i in range(1, num_images + 1):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image_path = os.path.join(save_dir, f"fake_image_{i}.jpg")
            with open(image_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: fake_image_{i}.jpg")
        else:
            print(f"Failed to fetch image {i}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {i}: {e}")

print("Download completed!")

