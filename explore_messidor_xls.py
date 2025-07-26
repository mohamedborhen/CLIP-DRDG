import pandas as pd

# 🔁 Change ce chemin pour tester chaque fichier
xls_path = r"C:\Users\ensi02\Desktop\dataset DR detection\datasets\messidor\base12.xls"

# Lire le fichier
df = pd.read_excel(xls_path)

# Afficher les colonnes et un aperçu
print("📄 Colonnes :", df.columns)
print("\n🔍 Premières lignes :")
print(df.head())
