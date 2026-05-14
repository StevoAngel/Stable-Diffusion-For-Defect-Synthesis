import gradio as gr
import torch
import os
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, DDPMScheduler
from peft import PeftModel

# ==============================================================================
# 1. Global Configuration and Model Loading
# ==============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/sd-controlnet-canny"
lora_path = "lora_weights" # Relative to app.py location

print(f"Status: Loading models on {device}...")

# Load ControlNet and base U-Net
controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
base_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

# Inject LoRA weights
unet = PeftModel.from_pretrained(base_unet, lora_path)
unet.eval()

# Build Inference Pipeline
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    unet=unet,
    controlnet=controlnet,
    scheduler=scheduler,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

pipe.safety_checker = None

# ==============================================================================
# 2. Logic: Latent Arithmetic & Synthesis
# ==============================================================================
def get_text_embeddings(prompt):
    """Extracts mathematical embeddings from text."""
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        return pipe.text_encoder(text_inputs.input_ids.to(device))[0]

def generate_defect(alpha, canny_image, seed):
    """
    Core function to synthesize the defect using latent interpolation.
    """
    # Define the two semantic poles
    prompt_ok = "photo of SKS_PART, top-down industrial inspection, circular machined component, flawless smooth surface, neutral factory lighting"
    prompt_def = "photo of SKS_PART, top-down industrial inspection, circular machined component, severe porosity defect, surface blowholes and cavities, neutral factory lighting"
    negative_prompt = "relic, face, skull, mask, bronze, jewelry, colorful, 3d render, blurry, distorted"

    # Compute Embeddings
    embed_ok = get_text_embeddings(prompt_ok)
    embed_def = get_text_embeddings(prompt_def)
    embed_neg = get_text_embeddings(negative_prompt)

    # LATENT ARITHMETIC: Linear interpolation between OK and Defect
    interp_embeds = (1.0 - alpha) * embed_ok + alpha * embed_def

    # Fixed seed for consistent background/texture
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    # Process image for ControlNet
    canny_image = canny_image.convert("RGB").resize((320, 320))

    # Generate
    output = pipe(
        prompt_embeds=interp_embeds,
        negative_prompt_embeds=embed_neg,
        image=canny_image,
        width=320,
        height=320,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    return output

# ==============================================================================
# 3. Gradio Interface Layout
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏭 Stable Diffusion Defect Synthesis
        ### Industrial Defect Generation via ControlNet & LoRA Latent Arithmetic
        This interface allows you to control the severity of defects defects in cast components using a progressive scale (α).
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_canny = gr.Image(label="ControlNet Canny Map", type="pil")
            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Defect Severity (α)")
            seed_val = gr.Number(value=42, label="Random Seed")
            btn = gr.Button("Synthesize Defect", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Synthesized Industrial Part")

    # Example Section
    gr.Examples(
        examples=[
            [0.0, "assets/sample_canny.jpeg", 42],
            [0.5, "assets/sample_canny.jpeg", 42],
            [1.0, "assets/sample_canny.jpeg", 42]
        ],
        inputs=[alpha_slider, input_canny, seed_val]
    )

    btn.click(fn=generate_defect, inputs=[alpha_slider, input_canny, seed_val], outputs=output_img)

if __name__ == "__main__":
    demo.launch()