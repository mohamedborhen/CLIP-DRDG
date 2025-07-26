import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Configuration ===
base_input_dir = r"put the path to your downloaded eyepacs dataset"
base_output_dir = "eyepacs_converted"
img_dir = os.path.join(base_input_dir, "train")

# === Lire les labels ===
csv_path = os.path.join(base_input_dir, "trainLabels.csv")
df = pd.read_csv(csv_path)

# === Diviser train/val/test ===
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["level"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["level"], random_state=42)

splits = [("train", train_df), ("valid", val_df), ("test", test_df)]

# === Créer les dossiers et copier ===
for split_name, split_df in splits:
    print(f"Processing {split_name}...")
    for _, row in split_df.iterrows():
        image_id = row["image"]
        label = str(row["level"])

        for ext in [".png", ".jpg", ".jpeg"]:
            src_img_path = os.path.join(img_dir, image_id + ext)
            if os.path.exists(src_img_path):
                break
        else:
            print(f"❌ Image not found: {image_id}")
            continue

        dst_dir = os.path.join(base_output_dir, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_img_path = os.path.join(dst_dir, image_id + ext)
        shutil.copy(src_img_path, dst_img_path)

print("✅ EyePACS converted and ready for CLIP-DRDG.")
