# Customer-Segmentation-Project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# STEP 1: GENERATE SYNTHETIC CUSTOMER DATA
# ==========================================
# Setting a random seed ensures the exact same data is generated every time
np.random.seed(42)
n_customers = 200

data = {
    'CustomerID': range(1001, 1001 + n_customers),
    'Age': np.random.randint(18, 70, size=n_customers),
    'Annual_Income_k$': np.random.randint(15, 130, size=n_customers),
    'Spending_Score_1_100': np.random.randint(1, 100, size=n_customers)
}

df = pd.DataFrame(data)
print("--- Step 1: Raw Customer Data Loaded (First 5 Rows) ---")
print(df.head())
print("\n" + "="*50 + "\n")

# ==========================================
# STEP 2: DATA PREPROCESSING & SCALING
# ==========================================
# Extracting features for clustering
features = ['Age', 'Annual_Income_k$', 'Spending_Score_1_100']
X = df[features]

# Standardize features (Mean = 0, Variance = 1) so larger ranges don't dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# STEP 3: THE ELBOW METHOD (To find optimal K)
# ==========================================
wcss = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method chart in a separate window/output
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method to Find Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True, alpha=0.5)
plt.show(block=False)  # block=False allows execution to continue to the next plot

# ==========================================
# STEP 4: TRAIN THE FINAL K-MEANS MODEL
# ==========================================
# Based on standard distribution, 4 clusters cleanly segments this specific data
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)

# Predict and add cluster labels to our original dataframe
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("--- Step 4: Data with Cluster Labels Assigned (First 5 Rows) ---")
print(df.head())
print("\n" + "="*50 + "\n")

# ==========================================
# STEP 5: PROFILING THE CLUSTERS (ANALYSIS)
# ==========================================
# Calculate averages for each segment
cluster_profile = df.groupby('Cluster')[features].mean().reset_index()

# Calculate how many customers belong to each segment
cluster_counts = df['Cluster'].value_counts().reset_index().rename(columns={'count': 'Customer_Count'})

# Combine averages and counts into a single summary table
final_profile = pd.merge(cluster_profile, cluster_counts, on='Cluster')

print("--- Step 5: Customer Segment Profiles (Averages & Counts) ---")
print(final_profile.to_string(index=False))
print("\n" + "="*50 + "\n")

# ==========================================
# STEP 6: VISUALIZE THE SEGMENTS
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Annual_Income_k$', 
    y='Spending_Score_1_100', 
    hue='Cluster', 
    data=df, 
    palette='Set1', 
    s=100, 
    alpha=0.85
)

plt.title('Customer Segments: Annual Income vs. Spending Score', fontsize=14)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
