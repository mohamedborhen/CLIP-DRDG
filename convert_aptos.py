import os
import shutil
import pandas as pd

# Set the path to your APTOS dataset folder
base_input_dir = r"put the path to your downloaded aptos dataset"
base_output_dir = "augmented_resized_V2"

splits = ["train", "valid", "test"]

csv_map = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv"
}

for split in splits:
    print(f"Processing {split}...")

    # Load CSV file
    csv_path = os.path.join(base_input_dir, csv_map[split])
    df = pd.read_csv(csv_path)

    # Folder containing the images
    img_dir = os.path.join(base_input_dir, split)

    for _, row in df.iterrows():
        image_id = row["id_code"]
        label = str(row["diagnosis"])

        # Try multiple extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            src_img_path = os.path.join(img_dir, image_id + ext)
            if os.path.exists(src_img_path):
                break
        else:
            print(f"Image not found: {image_id} with extensions png/jpg/jpeg")
            continue

        dst_dir = os.path.join(base_output_dir, split, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_img_path = os.path.join(dst_dir, image_id + ext)

        shutil.copy(src_img_path, dst_img_path)
