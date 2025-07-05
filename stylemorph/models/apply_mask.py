from PIL import Image
import numpy as np
import os

# --- CONFIGURATION ---
original_image_path = "stylemorph/test_images/1000089293.jpg"
mask_path = "stylemorph/outputs/01_mask.png"
masked_output_path = "stylemorph/outputs/01masked_full.jpg"
cropped_output_path = "stylemorph/outputs/01masked_cropped.jpg"

# --- LOAD IMAGES ---
image = Image.open(original_image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")  # Convert to grayscale

# Resize mask to match image if needed
if mask.size != image.size:
    mask = mask.resize(image.size, Image.BILINEAR)

# Convert mask to 0 or 1
mask_np = np.array(mask)
binary_mask = (mask_np > 128).astype(np.uint8)

# Apply mask
image_np = np.array(image)
masked_image_np = image_np * binary_mask[:, :, np.newaxis]

# Save full masked image
masked_image = Image.fromarray(masked_image_np)
masked_image.save(masked_output_path)

# --- CROP TO BOUNDING BOX ---
coords = np.argwhere(binary_mask)
if coords.size > 0:
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = masked_image.crop((x0, y0, x1, y1))
    cropped.save(cropped_output_path)
    print(f"[INFO] Cropped masked image saved to: {cropped_output_path}")
else:
    print("[WARNING] No mask detected to crop.")

print(f"[INFO] Full masked image saved to: {masked_output_path}")
