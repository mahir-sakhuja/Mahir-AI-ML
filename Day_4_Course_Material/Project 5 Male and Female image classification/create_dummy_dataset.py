import os
import cv2
import numpy as np

def create_dummy_images():
    base_dir = r"c:\Users\Mahir sakhuja\New folder (4)\Day 4 Course Material\Project 5 Male and Female image classification\dataset"
    male_dir = os.path.join(base_dir, "Male")
    female_dir = os.path.join(base_dir, "Female")
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)
    
    # Create 50 male images (vertical lines)
    for i in range(50):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add random noise/variations
        img += np.random.randint(0, 30, (64, 64, 3), dtype=np.uint8)
        # Draw vertical line
        cv2.line(img, (32 + np.random.randint(-2, 3), 5), (32 + np.random.randint(-2, 3), 59), (255, 200, 150), 3)
        cv2.imwrite(os.path.join(male_dir, f"male_{i}.jpg"), img)
        
    # Create 50 female images (horizontal lines)
    for i in range(50):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add random noise/variations
        img += np.random.randint(0, 30, (64, 64, 3), dtype=np.uint8)
        # Draw horizontal line
        cv2.line(img, (5, 32 + np.random.randint(-2, 3)), (59, 32 + np.random.randint(-2, 3)), (150, 200, 255), 3)
        cv2.imwrite(os.path.join(female_dir, f"female_{i}.jpg"), img)
    print("Dummy dataset created successfully!")

if __name__ == "__main__":
    create_dummy_images()
