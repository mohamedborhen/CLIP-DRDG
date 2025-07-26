import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Chemins à modifier selon ton dossier
base_dir = r"C:\Users\ensi02\Downloads\messidor-2"
images_dir = os.path.join(base_dir, "messidor-2")
csv_path = os.path.join(base_dir, "data.csv")  # remplace par le nom exact de ton csv

output_dir = "messidor2_converted"

# Charger le CSV
df = pd.read_csv(csv_path)

# Garde uniquement les images gradables avec label valide 0-4
df = df[df['adjudicated_gradable'] == 1]
df = df[df['diagnosis'].isin([0,1,2,3,4])]

# Ajouter chemin complet vers les images
df["image_path"] = df["id_code"].apply(lambda x: os.path.join(images_dir, x))

# Garder uniquement les images existantes
df = df[df["image_path"].apply(os.path.exists)]

# Split train/val/test (70/15/15 stratifié par label)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["diagnosis"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["diagnosis"], random_state=42)

splits = [("train", train_df), ("valid", val_df), ("test", test_df)]

# Copier les images dans la structure demandée
for split_name, split_df in splits:
    print(f"Processing {split_name}...")
    for _, row in split_df.iterrows():
        label = str(row["diagnosis"])
        src_path = row["image_path"]
        dst_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

print("✅ Messidor-2 prêt dans", output_dir)
