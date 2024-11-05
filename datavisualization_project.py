import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('customer_data.csv')

# Displaying basic info
print("Data Summary:")
print(data.head())
print(data.describe())
print(data.info())

# Data Preprocessing
data['LastPurchaseDate'] = pd.to_datetime(data['LastPurchaseDate'])
data['DaysSinceLastPurchase'] = (pd.Timestamp.now() - data['LastPurchaseDate']).dt.days

# Feature selection
features = data[['PurchaseAmount', 'BrowsingTime', 'DaysSinceLastPurchase']]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=0)  # Choosing 3 clusters
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualization
plt.figure(figsize=(12, 6))

# Scatter plot of Purchase Amount vs. Browsing Time colored by Cluster
plt.subplot(1, 2, 1)
sns.scatterplot(x='PurchaseAmount', y='BrowsingTime', hue='Cluster', data=data, palette='viridis')
plt.title('Purchase Amount vs. Browsing Time')
plt.xlabel('Purchase Amount')
plt.ylabel('Browsing Time')

# Distribution of Days Since Last Purchase
plt.subplot(1, 2, 2)
sns.histplot(data['DaysSinceLastPurchase'], kde=True, bins=30, color='blue')
plt.title('Distribution of Days Since Last Purchase')
plt.xlabel('Days Since Last Purchase')

plt.tight_layout()
plt.show()

# Save clustered data to new CSV
data.to_csv('clustered_customer_data.csv', index=False)
print("Clustered data saved to 'clustered_customer_data.csv'.")
