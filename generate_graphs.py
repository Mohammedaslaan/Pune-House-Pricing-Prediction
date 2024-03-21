import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the 'static' folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load the dataset
df = pd.read_csv("house_price_data.csv")  # Replace "house_price_data.csv" with your dataset filename

# Select only numeric columns for correlation matrix heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='area', y='price')
plt.title('Scatter plot of area vs. price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.savefig('static/scatter_plot.png')  # Save the generated scatter plot as an image file
plt.close()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='mainroad', y='price')
plt.title('Box plot of mainroad vs. price')
plt.xlabel('Main Road')
plt.ylabel('Price')
plt.savefig('static/box_plot.png')  # Save the generated box plot as an image file
plt.close()

# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='airconditioning', y='price')
plt.title('Average price based on airconditioning')
plt.xlabel('Air Conditioning')
plt.ylabel('Average Price')
plt.savefig('static/bar_plot.png')  # Save the generated bar plot as an image file
plt.close()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='area', bins=20, kde=True)
plt.title('Histogram of area')
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.savefig('static/histogram.png')  # Save the generated histogram as an image file
plt.close()

# Correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation matrix heatmap')
plt.savefig('static/correlation_heatmap.png')  # Save the generated correlation heatmap as an image file
plt.close()

# Pair plot
sns.pairplot(df, vars=['area', 'bedrooms', 'bathrooms', 'price'])
plt.title('Pair plot of area, bedrooms, bathrooms, and price')
plt.savefig('static/pair_plot.png')  # Save the generated pair plot as an image file
plt.close()

# Line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='area', y='price')
plt.title('Line plot of area vs. price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.savefig('static/line_plot.png')  # Save the generated line plot as an image file
plt.close()
