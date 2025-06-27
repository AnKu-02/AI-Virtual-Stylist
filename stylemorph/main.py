# main.py

from segmentation.u2net_infer import load_model, segment_person
import os

model_path = 'stylemorph/segmentation/u2net.pth'
input_image = 'stylemorph/test_images/1000089293.jpg'
output_mask = 'stylemorph/outputs/01_mask.png'

def main():
    os.makedirs('outputs', exist_ok=True)

    print("[INFO] Loading model...")
    model = load_model(model_path)

    print(f"[INFO] Segmenting {input_image}...")
    segment_person(input_image, output_mask, model)
    print(f"[DONE] Saved mask to {output_mask}")

if __name__ == "__main__":
    main()