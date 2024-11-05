import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:\\Users\\my\\Downloads\\ecommerce_sales_analysis.csv")
# Displaying Dataset Overview

# 1. Show the dimensions of the dataset
print("Dimensions of the dataset:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# 2. Show the column names and data types
print("Column Names and Data Types:")
print(df.dtypes)
print("\n")

# 3. Display basic statistics for numerical columns
print("Summary Statistics for Numerical Columns:")
print(df.describe())
print("\n")

# 4. Calculate and display the mean for specific columns
mean_price = df['price'].mean()
mean_review_score = df['review_score'].mean()
mean_review_count = df['review_count'].mean()

print("Mean Values:")
print(f"Mean Price: {mean_price:.2f}")
print(f"Mean Review Score: {mean_review_score:.2f}")
print(f"Mean Review Count: {mean_review_count:.2f}")
print("\n")

# 5. Checking for missing values in each column
print("Missing Values per Column:")
print(df.isnull().sum())
print("\n")

# 6. Checking for unique values in categorical columns
print("Unique Categories in 'category':")
print(df['category'].unique())


# Data Cleaning and Transformation Steps

# 1. Handling Missing Values: Drop rows with missing 'product_id' (assumed essential) and fill missing 'product_name' values with a placeholder
df = df.dropna(subset=['product_id'])
df['product_name'].fillna('Unknown Product', inplace=True)

# 2. Removing Duplicates: Drop any duplicate rows
df = df.drop_duplicates()

# 3. Data Type Conversion: Convert monthly sales columns to numeric
sales_columns = [col for col in df.columns if 'sales_month' in col]
df[sales_columns] = df[sales_columns].apply(pd.to_numeric, errors='coerce')

# 4. Outlier Detection: Identify and handle outliers in 'price' using IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

# 5. Standardizing Categorical Data: Ensure 'category' values are consistent by standardizing text format
df['category'] = df['category'].str.strip().str.lower()

# Graphical Representations

# 1. Sales Over Time: Line plot of monthly sales
monthly_sales = df[sales_columns].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o', color='b')
plt.title('Sales Over Time')
plt.xlabel('Months')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# 2. Category Popularity: Bar chart for sales across categories
category_sales = df.groupby('category')[sales_columns].sum().sum(axis=1).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar', color='teal')
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()

# 3. Price Distribution: Histogram and box plot for 'price'
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=20, kde=True, color='purple')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='price', data=df, color='skyblue')
plt.title('Box Plot of Price Distribution')
plt.xlabel('Price')
plt.show()

# 4. Correlation Matrix: Heatmap for numerical features
# Select numerical columns for correlation matrix (excluding sales_month columns if needed)
numerical_cols = ['price', 'review_score', 'review_count'] + sales_columns
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting area
plt.figure(figsize=(15, 10))

# 1. Price Distribution Histogram
plt.subplot(2, 2, 1)
plt.hist(df['price'], bins=30, color='skyblue', edgecolor='black')
plt.title('Price Distribution Histogram')
plt.xlabel('Price')
plt.ylabel('Frequency')

# 2. Price Scatter Plot (to observe any price and review score relationship, if applicable)
plt.subplot(2, 2, 2)
plt.scatter(df['price'], df['review_score'], alpha=0.5, color='orange')
plt.title('Price vs. Review Score')
plt.xlabel('Price')
plt.ylabel('Review Score')

# 3. Price Box Plot
plt.subplot(2, 2, 3)
sns.boxplot(x=df['price'], color='lightgreen')
plt.title('Box Plot of Price')
plt.xlabel('Price')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
