import torch
import os
import warnings
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

def setup_output_directory(base_path="generated_faces_hq"):
    os.makedirs(base_path, exist_ok=True)
    return base_path

def initialize_model():
    model_id = "stabilityai/stable-diffusion-2-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    return pipe

def get_diverse_prompt():
    age_groups = [
        "young adult (25-35 years old)",
        "middle-aged (35-50 years old)",
        "mature adult (50-65 years old)"
    ]
    
    genders = ["male", "female"]
    facial_features = [
        "sharp features", "soft features", "defined jawline", "high cheekbones", "warm smile", "gentle eyes"
    ]
    skin_tones = [
        "medium brown skin tone", "deep brown skin tone", "light brown skin tone", "olive skin tone"
    ]
    photo_styles = [
        "close-up headshot from top of hair to neck, no shoulders visible",
        "portrait headshot from top of hair to neck, no shoulders visible",
        "studio headshot from top of hair to neck, no shoulders visible",
        "editorial headshot from top of hair to neck, no shoulders visible"
    ]
    lighting = [
        "studio lighting setup", "natural window lighting", "professional three-point lighting", "soft diffused lighting"
    ]
    quality_modifiers = [
        "8k resolution", "highly detailed", "professional photography", "sharp focus", "canon EOS R5", "award-winning portrait"
    ]

    prompt_elements = [
        f"{random.choice(photo_styles)} of an Indian {random.choice(genders)}, {random.choice(age_groups)},",
        f"{random.choice(skin_tones)}, {random.choice(facial_features)},",
        f"{random.choice(lighting)}, {', '.join(random.sample(quality_modifiers, 2))}"
    ]
    
    prompt = " ".join(prompt_elements)
    negative_prompt = (
        "cartoon, anime, illustration, painting, drawing, artwork, dark shadows, "
        "deformed features, blurry, unrealistic, distorted, bad anatomy, "
        "partial face, half face, cropped face, shoulders, upper body"
    )

    return prompt, negative_prompt

def generate_faces(pipe, num_images=10000, output_dir="generated_faces_hq"):
    for i in tqdm(range(num_images), desc="Generating images"):
        image_path = os.path.join(output_dir, f"indian_face_hq_{i:05d}.png")
        
        if os.path.exists(image_path):
            continue  # Skip existing images
        
        prompt, negative_prompt = get_diverse_prompt()
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=40,
                guidance_scale=7.0,
                width=768,
                height=768
            ).images[0]
        
        image.save(image_path, "PNG", compress_level=0)

def main():
    print("Setting up Stable Diffusion 2.1 face generation pipeline (GPU if available)...")
    output_dir = setup_output_directory()
    pipe = initialize_model()
    print("Starting generation of missing face + neck images...")
    print(f"Images will be saved to: {output_dir}")
    generate_faces(pipe, num_images=10000, output_dir=output_dir)
    print("Generation complete!")

if __name__ == "__main__":
    main()
