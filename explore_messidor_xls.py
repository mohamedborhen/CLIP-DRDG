import pandas as pd

# 🔁 Change ce chemin pour tester chaque fichier
xls_path = r"put the path to messidor's base12.xls here"

# Lire le fichier
df = pd.read_excel(xls_path)

# Afficher les colonnes et un aperçu
print("📄 Colonnes :", df.columns)
print("\n🔍 Premières lignes :")
print(df.head())
