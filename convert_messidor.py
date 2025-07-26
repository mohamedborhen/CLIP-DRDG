import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Bases Messidor à traiter ===
bases = ["base11", "base12", "base13", "base14"]
base_dir = r"put the path to your downloaded messidor dataset"
output_dir = "messidor_converted"

# === Lire et fusionner tous les fichiers .xls ===
all_data = []

for base in bases:
    xls_path = os.path.join(base_dir, f"{base}.xls")
    img_dir = os.path.join(base_dir, base)
    
    df = pd.read_excel(xls_path)
    df["image_path"] = df["Image name"].apply(lambda x: os.path.join(img_dir, x.strip()))
    df["label"] = df["Retinopathy grade"]
    
    all_data.append(df[["image_path", "label"]])

full_df = pd.concat(all_data, ignore_index=True)

# === Nettoyer les chemins d'image ===
full_df = full_df[full_df["label"].isin([0,1,2,3,4])]
full_df = full_df[full_df["image_path"].apply(os.path.exists)]

# === Split en train/val/test ===
train_df, temp_df = train_test_split(full_df, test_size=0.3, stratify=full_df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

splits = [("train", train_df), ("valid", val_df), ("test", test_df)]

# === Copier les images ===
for split_name, split_df in splits:
    print(f"Processing {split_name}...")
    for _, row in split_df.iterrows():
        label = str(row["label"])
        src_path = row["image_path"]
        ext = os.path.splitext(src_path)[1]
        
        dst_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

print("✅ Conversion terminée : messidor_converted/")
