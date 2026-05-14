import os
import shutil
import cv2
import numpy as np
from PIL import Image

# Origin data paths:
ok_path_raw = "../data/raw/casting/casting_512x512/ok_front/"
defect_path_raw = "../data/raw/casting/casting_512x512/def_front/"

# Destination paths for images, captions AND CANNY edges:
TARGET_SIZE = 256 # Change to 512 if you prefer slow but ultra-hi-res training

base_processed = f"../data/processed/casting/casting_{TARGET_SIZE}x{TARGET_SIZE}/"
ok_path_processed = os.path.join(base_processed, "ok_front/images/")
ok_canny_processed = os.path.join(base_processed, "ok_front/canny/")

defect_path_processed = os.path.join(base_processed, "def_front/images/")
defect_canny_processed = os.path.join(base_processed, "def_front/canny/")

# Captions (Implicit string concatenation for a single perfect line):
caption_ok = (
    "photo of SKS_PART, top-down industrial inspection, circular machined component, "
    "flat grey ferrous material, flawless smooth surface, QC passed, neutral factory lighting, "
    "plain background"
)

caption_defect = (
    "photo of SKS_PART, top-down industrial inspection, circular machined component, "
    "flat grey ferrous material, severe porosity defect, surface blowholes and cavities, "
    "QC failed manufacturing reject, neutral factory lighting, plain background"
)

def create_advanced_dataset(raw_path, processed_img_path, processed_canny_path, caption, size):
    """
    1. Resizes original images.
    2. Creates Canny Edge maps.
    3. Saves everything alongside perfect captions.
    """
    # Create clean directories
    os.makedirs(processed_img_path, exist_ok=True)
    os.makedirs(processed_canny_path, exist_ok=True)

    images = [f for f in os.listdir(raw_path) if f.endswith(('.jpeg'))]
    counter = 0

    for image_name in images:
        origin_img_path = os.path.join(raw_path, image_name)
        
        # --- 1. RESIZE IMAGE ---
        # We use PIL for high-quality resizing (LANCZOS)
        img_pil = Image.open(origin_img_path).convert("RGB")
        img_resized = img_pil.resize((size, size), Image.Resampling.LANCZOS)
        
        # Save the resized RGB image
        destination_img = os.path.join(processed_img_path, image_name)
        img_resized.save(destination_img)

        # --- 2. GENERATE CANNY EDGE MAP ---
        # Convert PIL to OpenCV format (Numpy array, BGR)
        img_cv = np.array(img_resized)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Apply Canny algorithm (Lower and Upper thresholds)
        # 100 and 200 are standard, but you can tweak them if the lines are too thick/thin
        edges = cv2.Canny(img_cv, 100, 200) 
        
        # Save the Canny image
        destination_canny = os.path.join(processed_canny_path, image_name)
        cv2.imwrite(destination_canny, edges)

        # --- 3. CREATE CAPTION (.txt) ---
        filename_base = os.path.splitext(image_name)[0]
        caption_path = os.path.join(processed_img_path, f"{filename_base}.txt")

        with open(caption_path, "w") as f:
            f.write(caption)

        counter += 1

    print(f"Processed {counter} items -> {processed_img_path} (Images, Canny & Captions)")

# --- EXECUTION ---
print(f"Building Dataset at {TARGET_SIZE}x{TARGET_SIZE} resolution...")

print("\nProcessing OK Dataset...")
create_advanced_dataset(ok_path_raw, ok_path_processed, ok_canny_processed, caption_ok, TARGET_SIZE)

print("\nProcessing Defective Dataset...")
create_advanced_dataset(defect_path_raw, defect_path_processed, defect_canny_processed, caption_defect, TARGET_SIZE)

print("\nDataset v2 completely generated and optimized for ControlNet!")