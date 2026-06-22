import os
import numpy as np
import random
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, DDPMScheduler
from peft import PeftModel
from tqdm.auto import tqdm

# =====================================
# 1. Random seed to ensure reproducibility
# =====================================
def set_seed(seed=42):
    # 1. Native python seed:
    random.seed(seed)

    # 2. Evironment python seed:
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Numpy seed:
    np.random.seed(seed)
    
    # 4. PyTorch seed (CPU)
    torch.manual_seed(seed)
    
    # 5. PyTorch seed (GPU / CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si usas múltiples GPUs
    
    # 6. CuDNN deterministic for stability in math operations:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) # Call seed function

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/sd-controlnet-canny"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Base project/root dir (one level up from this script)
lora_path = os.path.abspath(os.path.join(ROOT_DIR, "demo", "lora_weights"))
TARGET_SIZE = 256 # Output image size
NUM_IMAGES = 1000 # Number of images to generate


# ==============================================================================
# 2. Paths
# ==============================================================================

# Input paths (Real Canny Maps to guide the generation):
canny_dir = os.path.join(ROOT_DIR, f"data/processed/casting/casting_{TARGET_SIZE}x{TARGET_SIZE}/ok_front/canny")

# Output paths (absolute):
output_dir_ok = os.path.join(ROOT_DIR, "data/processed/generated_casting/sd_ok")
output_dir_def = os.path.join(ROOT_DIR, "data/processed/generated_casting/sd_def")
os.makedirs(output_dir_ok, exist_ok=True)
os.makedirs(output_dir_def, exist_ok=True)

# Validate key paths early with helpful errors
if not os.path.isdir(canny_dir):
    raise FileNotFoundError(f"Canny directory not found: {canny_dir}")
if not os.path.isdir(lora_path):
    raise FileNotFoundError(f"LoRA weights directory not found: {lora_path}")
if not any(p.endswith("adapter_config.json") for p in os.listdir(lora_path)) and not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
    # provide a clear message but don't try to auto-fix
    raise FileNotFoundError(f"adapter_config.json not found in LoRA directory: {lora_path}")


# ==============================================================================
# 3. Load Models
# ==============================================================================

print("Loading Stable Diffusion with ControlNet...")
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(device)
base_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to(device)

print("Applying LoRA fine-tuning weights...")
unet = PeftModel.from_pretrained(base_unet, lora_path, torch_dtype=torch.float16).to(device)
unet.eval()  # Set to evaluation mode

scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float16
).to(device)

pipe.safety_checker = None # Disable safety checker for industrial images
pipe.set_progress_bar_config(disable=False) # Disable tqdm progress bars to avoid spamming the console


# ==============================================================================
# 4. Conditioning Prompts
# ==============================================================================

prompt_ok = (
    "photo of SKS_PART, top-down industrial inspection, circular machined component, "
    "flat grey ferrous material, flawless smooth surface, QC passed, neutral factory lighting, "
    "plain background"
)

prompt_def = (
    "photo of SKS_PART, top-down industrial inspection, circular machined component, "
    "flat grey ferrous material, severe porosity defect, surface blowholes and cavities, "
    "QC failed manufacturing reject, neutral factory lighting, plain background"
)

negative_prompt = (
    "relic, antique, museum, face, skull, mask, statue, bronze, gold, jewelry, "
    "ornate, complex shape, intricate, art, shiny, colorful, 3d render"
)


# ==============================================================================
# 5. Image Generation Loop
# ==============================================================================

# Get list of Canny maps to use as conditioning inputs (support common extensions)
canny_files = [f for f in os.listdir(canny_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if len(canny_files) == 0:
    raise FileNotFoundError(f"No image files found in canny directory: {canny_dir}")

print("Starting image generation...")

base_seed = 42  # Base seed for reproducibility

for i in tqdm(range(NUM_IMAGES), desc="Generating SD Images"):
    # Select a Canny Map (cycling through the available ones)
    canny_file = canny_files[i % len(canny_files)]
    canny_path = os.path.join(canny_dir, canny_file)
    canny_image = Image.open(canny_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)

    # Set a unique seed for each image to ensure diversity while maintaining reproducibility
    image_seed = base_seed + i
    g_cuda = torch.Generator(device).manual_seed(image_seed)

    # Generate Ok image:
    img_ok = pipe(
        prompt=prompt_ok,
        negative_prompt=negative_prompt,
        image=canny_image,
        width=TARGET_SIZE,
        height=TARGET_SIZE,
        num_inference_steps=20, # Fewer steps for faster generation
        guidance_scale=7.5, # Default guidance scale
        generator=g_cuda
    ).images[0]
    img_ok.save(os.path.join(output_dir_ok, f"ok_{i:04d}.jpeg"))

    # Generate Defective image:
    img_def = pipe(
        prompt=prompt_def,
        negative_prompt=negative_prompt,
        image=canny_image,
        width=TARGET_SIZE,
        height=TARGET_SIZE,
        num_inference_steps=20, # Fewer steps for faster generation
        guidance_scale=7.5, # Default guidance scale
        generator=g_cuda
    ).images[0]
    img_def.save(os.path.join(output_dir_def, f"def_{i:04d}.jpeg"))

print("Image generation completed!")