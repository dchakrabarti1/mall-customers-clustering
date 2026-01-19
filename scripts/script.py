import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, SpectralClustering


DATA_PATH = "Mall_Customers.csv"

df = pd.read_csv(DATA_PATH)
age_bins = [0,25,40,60,100]

age_labels = ['Age_<=25', 'Age_26_40', 'Age_41_60', 'Age_>60']
df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
age_dummies = pd.get_dummies(df['AgeBin'], drop_first=False)

X_continuous = df[['Annual Income (k$)', 'Spending Score (1-100)']]

X = pd.concat([X_continuous, age_dummies], axis=1)

scaler= StandardScaler()

X_scaled_cont = scaler.fit_transform(X_continuous)
X_scaled = np.concatenate([X_scaled_cont, age_dummies.values], axis=1)

income = df['Annual Income (k$)']
spending = df['Spending Score (1-100)']

def summarize_clusters(df, labels, label_name):
    tmp = df.copy()
    tmp[label_name] = labels

    print(f"\n===== Cluster summary: {label_name} =====")
    print("Cluster sizes:")
    print(tmp[label_name].value_counts().sort_index())
    print()

    summary = tmp.groupby(label_name)[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].agg(['mean', 'std', 'min', 'max'])
    print("Summary stats (Age, Income, Spending):")
    print(summary)
    print("========================================\n")

# 2. Silhouette statistic to choose ksil

def compute_silhouette_scores(X, k_range):
    sil_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)
        print(f"k = {k}, silhouette score = {score:.4f}")
    return sil_scores

k_range = range(2, 11)

print("Computing silhouette scores...")
sil_scores = compute_silhouette_scores(X_scaled, k_range)

plt.figure()
plt.plot(list(k_range), sil_scores, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette score')
plt.title('Silhouette scores for different k (K-Means)')
plt.grid(True)
plt.show()

ksil = k_range[int(np.argmax(sil_scores))]
print(f"\nBest k by silhouette (ksil) = {ksil}\n")

# 3. Gap statistic to choose kgap
def kmeans_Wk(X, n_clusters, n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    kmeans.fit(X)
    return kmeans.inertia_

def gap_statistic(X, k_range, B=20):
    X = np.array(X)
    n, d = X.shape

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    gaps = []
    sk_vals = []

    for k in k_range:
        print(f"Computing Gap statistic for k = {k}...")
        Wk = kmeans_Wk(X, k)

        Wk_refs = np.zeros(B)
        for b in range(B):
            X_ref = np.random.uniform(mins, maxs, size=(n, d))
            Wk_refs[b] = kmeans_Wk(X_ref, k)

        logWk = np.log(Wk)
        logWk_refs = np.log(Wk_refs)

        gap = np.mean(logWk_refs) - logWk
        s_k = np.std(logWk_refs) * np.sqrt(1 + 1.0 / B)

        gaps.append(gap)
        sk_vals.append(s_k)

    return np.array(gaps), np.array(sk_vals)

print("Computing Gap statistics...")
gaps, sk_vals = gap_statistic(X_scaled, k_range, B=20)

plt.figure()
plt.plot(list(k_range), gaps, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Gap statistic')
plt.title('Gap statistic for different k (K-Means)')
plt.grid(True)
plt.show()

kgap = None
for i in range(len(k_range) - 1):
    if gaps[i] >= gaps[i+1] - sk_vals[i+1]:
        kgap = k_range[i]
        break

if kgap is None:
    kgap = k_range[int(np.argmax(gaps))]

print(f"\nBest k by Gap statistic (kgap) = {kgap}\n")

# 4. K-Means for kgap and ksil + plots + summaries

# KMeans for kgap
kmeans_gap = KMeans(n_clusters=kgap, n_init=10, random_state=42)
labels_gap = kmeans_gap.fit_predict(X_scaled)

plt.figure()
scatter = plt.scatter(income, spending, c=labels_gap, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.title(f'K-Means Clusters (k = {kgap}, Gap Statistic)')
plt.colorbar(scatter, label='Cluster label')
plt.grid(True)
plt.show()

summarize_clusters(df, labels_gap, f'kmeans_gap_k{kgap}')

# KMeans for ksil
kmeans_sil = KMeans(n_clusters=ksil, n_init=10, random_state=42)
labels_sil = kmeans_sil.fit_predict(X_scaled)

plt.figure()
scatter = plt.scatter(income, spending, c=labels_sil, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.title(f'K-Means Clusters (k = {ksil}, Silhouette)')
plt.colorbar(scatter, label='Cluster label')
plt.grid(True)
plt.show()

summarize_clusters(df, labels_sil, f'kmeans_sil_k{ksil}')

# 5. Hierarchical clustering (Ward) + dendrogram + stats

print("Performing hierarchical clustering (Ward)...")
Z = linkage(X_scaled, method='ward')

# i. Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(
    Z,
    truncate_mode='level', 
    p=5
)
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.show()

k_hier = ksil
print(f"\nUsing k_hier = {k_hier} clusters for hierarchical clustering.\n")

hier_labels = fcluster(Z, t=k_hier, criterion='maxclust')

plt.figure()
scatter = plt.scatter(income, spending, c=hier_labels, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.title(f'Hierarchical Clustering (Ward, k = {k_hier})')
plt.colorbar(scatter, label='Cluster label')
plt.grid(True)
plt.show()

summarize_clusters(df, hier_labels, f'hierarchical_k{k_hier}')

print("Script finished.")
print("Now compare:")
print("- K-Means clusters (for kgap and ksil): shapes, separation, and summaries.")
print("- Hierarchical clusters (k_hier): dendrogram structure, scatter plot, and summaries.")


# 1. Generate Two Moons Data
np.random.seed(598)     
n = 200               
n_per_moon = n // 2     
r = 2                   
noise_var = 0.05
noise_sd = np.sqrt(noise_var)

theta1 = np.random.uniform(0, np.pi, n_per_moon)
x1 = r * np.cos(theta1) + np.random.normal(0, noise_sd, n_per_moon)
y1 = r * np.sin(theta1) + np.random.normal(0, noise_sd, n_per_moon)
labels1 = np.zeros(n_per_moon, dtype=int)

theta2 = np.random.uniform(0, np.pi, n_per_moon)
x2 = 1 - 2 * np.cos(theta2) + np.random.normal(0, noise_sd, n_per_moon)
y2 = -2 * np.sin(theta2) - 0.5 + np.random.normal(0, noise_sd, n_per_moon)
labels2 = np.ones(n_per_moon, dtype=int)

X = np.vstack([
    np.column_stack([x1, y1]),
    np.column_stack([x2, y2])
])
y_true = np.concatenate([labels1, labels2])

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="bwr", s=25)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two Moons Synthetic Data (True Labels)")
plt.axis("equal")
plt.tight_layout()
plt.show()
# 3. K-Means Clustering (k = 2)

kmeans = KMeans(n_clusters=2, random_state=598, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap="bwr", s=25)
plt.xlabel("x")
plt.ylabel("y")
plt.title("K-Means Clustering (k = 2)")
plt.axis("equal")
plt.tight_layout()
plt.show()

# 4. Spectral Clustering (k = 2)

spectral = SpectralClustering(
    n_clusters=2,
    affinity='rbf',       
    gamma=1.0,           
    random_state=598,
    assign_labels='kmeans' 
)
spectral_labels = spectral.fit_predict(X)

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap="bwr", s=25)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Spectral Clustering (k = 2, RBF Kernel)")
plt.axis("equal")
plt.tight_layout()
plt.show()