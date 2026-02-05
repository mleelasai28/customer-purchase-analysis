import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load All Datasets
# -------------------------------
orders = pd.read_csv("data/olist_orders_dataset.csv")
customers = pd.read_csv("data/olist_customers_dataset.csv")
items = pd.read_csv("data/olist_order_items_dataset.csv")
products = pd.read_csv("data/olist_products_dataset.csv")
payments = pd.read_csv("data/olist_order_payments_dataset.csv")
reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")
categories = pd.read_csv("data/product_category_name_translation.csv")

print("✅ All datasets loaded successfully")

# -------------------------------
# 2. Data Cleaning
# -------------------------------
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp'], errors='coerce'
)

# -------------------------------
# 3. Merge Datasets
# -------------------------------
df = orders.merge(customers, on='customer_id', how='left') \
           .merge(items, on='order_id', how='left') \
           .merge(products, on='product_id', how='left') \
           .merge(categories, on='product_category_name', how='left') \
           .merge(payments, on='order_id', how='left') \
           .merge(reviews, on='order_id', how='left')   # 👈 ADD THIS


print("✅ Data merged successfully")

# -------------------------------
# 4. Feature Engineering
# -------------------------------
df['TotalAmount'] = df['price'] + df['freight_value']
df['Month'] = df['order_purchase_timestamp'].dt.month
df['Year'] = df['order_purchase_timestamp'].dt.year

# -------------------------------
# 5. Dataset Overview
# -------------------------------
print("\n📊 Dataset Info:")
print(df.info())

print("\n📌 Sample Data:")
print(df.head())

# -------------------------------
# 6. Top Selling Product Categories
# -------------------------------
top_categories = (
    df.groupby('product_category_name_english')['TotalAmount']
    .sum()
    .sort_values(ascending=False)
)

print("\n🔥 Top Product Categories:")
print(top_categories.head(10))

plt.figure(figsize=(10, 5))
sns.barplot(
    x=top_categories.head(10).values,
    y=top_categories.head(10).index,
    palette="viridis"
)
plt.title("Top 10 Product Categories by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Monthly Sales Trend
# -------------------------------
monthly_sales = df.groupby('Month')['TotalAmount'].sum()

plt.figure(figsize=(8, 4))
sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker="o"
)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# -------------------------------
# 8. Customer Spending Analysis
# -------------------------------
top_customers = (
    df.groupby('customer_id')['TotalAmount']
    .sum()
    .sort_values(ascending=False)
)

print("\n💰 Top 10 Customers:")
print(top_customers.head(10))

# -------------------------------
# 9. Review Score Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='review_score', data=df)
plt.title("Customer Review Distribution")
plt.xlabel("Review Score")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 10. Save Cleaned Dataset
# -------------------------------
df.to_csv("cleaned_customer_data.csv", index=False)

print("\n✅ Cleaned dataset saved as 'cleaned_customer_data.csv'")
