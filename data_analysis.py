import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Load the dataset
# Ensure All_Diets.csv is in the same folder
df = pd.read_csv('All_Diets.csv')

# 2. Data Cleaning
# Handle missing values by filling in with the mean
df[['Protein(g)', 'Carbs(g)', 'Fat(g)' ]] = df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].fillna(df[['Protein(g)', 'Carbs(g)', 'Fat(g)']].mean())

# Standardize Diet_type names for clean grouping
df['Diet_type'] = df['Diet_type'].str.strip().str.capitalize()

# 3. Calculation
# Average macros per diet
avg_macros = df.groupby('Diet_type')[['Protein(g)', 'Carbs(g)', 'Fat(g)']].mean()

# Top 5 protein-rich recipes per diet
top_protein = df.sort_values('Protein(g)', ascending=False).groupby('Diet_type').head(5)

# New Metrics: Ratios (adding 1e-6 to avoid division by zero)
df['Proteins_to_Carbs_ratio'] = df['Protein(g)'] / (df['Carbs(g)'] + 1e-6)
df['Carbs_to_Fat_ratio'] = df['Carbs(g)'] / (df['Fat(g)'] + 1e-6)

# VISUALS

# 1. Bar Chart: Average Macronutrients
plt.figure(figsize=(10, 6))
avg_macros.plot(kind='bar', ax=plt.gca())
plt.title('Average Protein, Carbs, and Fat by Diet Type')
plt.ylabel('Grams (g)')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.show()

# 2. Heatmap: Relationship between macros and diets
plt.figure(figsize=(8, 6))
sns.heatmap(avg_macros, annot=True, cmap='YlGnBu')
plt.title('Macronutrient Intensity Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# 3. Scatter Plot: Top 5 Protein Recipes vs Cuisines
plt.figure(figsize=(12,6))
sns.scatterplot(data=top_protein, x='Diet_type', y='Protein(g)', hue='Cuisine_type', s=100)
plt.title('Top 5 Protein-Rich Recipes by Diet & Cuisine')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.show()

print("Task 1 Complete. Images saved to current folder.")