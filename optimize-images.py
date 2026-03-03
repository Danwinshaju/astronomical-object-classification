import os
from PIL import Image
from tqdm import tqdm

# =============================
# CONFIGURATION
# =============================
INPUT_DIR = "C:/Users/Rayon/Desktop/astronomical_object_classification/data/images/test/"       # Change this
OUTPUT_DIR = "./dataset_resized"     # Output folder
TARGET_SIZE = 224                  # 224x224 recommended

# =============================
# Helper: Resize with Padding
# =============================
def resize_with_padding(img, target_size):
    img.thumbnail((target_size, target_size), Image.LANCZOS)

    new_img = Image.new("RGB", (target_size, target_size))
    paste_x = (target_size - img.width) // 2
    paste_y = (target_size - img.height) // 2

    new_img.paste(img, (paste_x, paste_y))
    return new_img


# =============================
# Count Total Images First
# =============================
def count_images(input_dir):
    count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                count += 1
    return count


# =============================
# Main Processing Function
# =============================
def process_images():
    total_images = count_images(INPUT_DIR)
    print(f"\n🔍 Found {total_images} images.\n")

    if total_images == 0:
        print("⚠ No images found.")
        return

    with tqdm(total=total_images, desc="Resizing Images", unit="img") as pbar:

        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

                    input_path = os.path.join(root, file)

                    # Preserve folder structure
                    relative_path = os.path.relpath(root, INPUT_DIR)
                    output_folder = os.path.join(OUTPUT_DIR, relative_path)
                    os.makedirs(output_folder, exist_ok=True)

                    output_path = os.path.join(output_folder, file)

                    try:
                        with Image.open(input_path) as img:
                            img = img.convert("RGB")
                            img = resize_with_padding(img, TARGET_SIZE)
                            img.save(output_path)

                    except Exception as e:
                        print(f"\n❌ Skipped {file}: {e}")

                    pbar.update(1)

    print("\n✅ All images resized successfully!\n")


# =============================
# Run
# =============================
if __name__ == "__main__":
    process_images()