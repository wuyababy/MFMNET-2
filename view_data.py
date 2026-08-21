import pandas as pd

# 读取数据文件
print("Reading all_ddi.csv...")
ddi = pd.read_csv("data/raw/all_ddi.csv")
print("DDI shape:", ddi.shape)
print("DDI columns:", ddi.columns.tolist())
print("DDI first 5 rows:")
print(ddi.head(5))
print("\nDDI label distribution:")
print(ddi['label'].value_counts())

print("\nReading all_drug_embedding.csv...")
drug_embedding = pd.read_csv("data/raw/all_drug_embedding.csv")
print("Drug embedding shape:", drug_embedding.shape)
print("Drug embedding first 5 rows (first 10 columns):")
print(drug_embedding.iloc[:5, :10])

print("\nReading gene_vector.csv...")
gene_vector = pd.read_csv("data/raw/gene_vector.csv", header=None)
print("Gene vector shape:", gene_vector.shape)
print("Gene vector first 5 rows (first 10 columns):")
print(gene_vector.iloc[:5, :10])
