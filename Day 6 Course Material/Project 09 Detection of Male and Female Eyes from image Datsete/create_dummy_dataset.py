"""
Project 09 – Create Dummy Eye Dataset
======================================
Generates synthetic 64x64 grayscale "eye" images for male/female classification.
Male eyes → rounder shapes with thicker brows
Female eyes → almond shapes with thinner brows and longer lashes
"""
import os
import numpy as np
from PIL import Image, ImageDraw

def create_eye_image(gender, size=64):
    """Generate a synthetic eye image for the given gender."""
    img = Image.new("L", (size, size), color=200)  # light gray background
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    if gender == "male":
        # Rounder eye shape
        draw.ellipse([cx-18, cy-8, cx+18, cy+8], fill=255, outline=0, width=2)
        # Thicker eyebrow
        draw.arc([cx-20, cy-22, cx+20, cy-6], start=200, end=340, fill=0, width=3)
        # Pupil
        draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill=0)
        # Iris ring
        draw.ellipse([cx-8, cy-8, cx+8, cy+8], outline=80, width=1)
    else:
        # Almond-shaped eye
        draw.polygon([(cx-22, cy), (cx, cy-10), (cx+22, cy), (cx, cy+8)],
                     fill=255, outline=0)
        # Thinner eyebrow with arch
        draw.arc([cx-22, cy-26, cx+22, cy-8], start=195, end=345, fill=0, width=1)
        # Longer lashes (lines radiating from top)
        for angle_offset in range(-15, 16, 5):
            lx = cx + angle_offset
            draw.line([(lx, cy-9), (lx, cy-16)], fill=0, width=1)
        # Pupil
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=0)
        # Iris ring
        draw.ellipse([cx-7, cy-7, cx+7, cy+7], outline=60, width=1)

    # Add some noise for realism
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "dataset")
    categories = {"male": 100, "female": 100}

    for gender, count in categories.items():
        folder = os.path.join(base_dir, gender)
        os.makedirs(folder, exist_ok=True)

        for i in range(count):
            img = create_eye_image(gender)
            img.save(os.path.join(folder, f"{gender}_{i:04d}.png"))

        print(f"✅ Created {count} {gender} eye images → {folder}")

    print(f"\n📁 Dataset ready at: {base_dir}")


if __name__ == "__main__":
    main()
