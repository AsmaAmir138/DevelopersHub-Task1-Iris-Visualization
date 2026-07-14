import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = sns.load_dataset('iris')

# 2. Inspect Dataset
print("--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Summary Statistics ---")
print(df.describe())

# 3. Visualizations
sns.set_theme(style="whitegrid")

# Pairplot (Scatter plots)
sns.pairplot(df, hue="species", palette="Set2")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.savefig('iris_pairplot.png') # Graph save karne ke liye
plt.show()

# Histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = df.columns[:-1]
for i, col in enumerate(features):
    ax = axes[i//2, i%2]
    sns.histplot(data=df, x=col, hue="species", kde=True, ax=ax, palette="Set2", multiple="stack")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig('iris_histograms.png')
plt.show()
