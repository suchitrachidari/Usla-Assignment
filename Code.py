"""
Ride Safety Dataset - Clustering Analysis
Algorithms: K-Means (main), Hierarchical (Ward), DBSCAN, Gaussian Mixture
Visualisation: PCA 2-D projection, elbow/silhouette, cluster profile heatmap
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv("ride_safety_dataset.csv")
print("Shape:", df.shape)

# ----------------------------------------------------------------------
# 2. SELECT FEATURES  (the "factors" used to form clusters)
# ----------------------------------------------------------------------
# Only numeric features that describe how SAFE / RISKY a ride situation is.
# IDs, dates, coordinates and raw text columns are excluded on purpose.
features = [
    # Crime outcome factors
    "Crime_Severity",
    "Police_Response_Time_Minutes",
    "Resolved",
    "Suspect_Arrested",
    # Safety infrastructure factors
    "Nearby_CCTV_Cameras",
    "Num_Police_Stations_Nearby",
    "Public_Complaints",
    # Road / traffic factors
    "Road_Accidents_Reported",
    "Traffic_Congestion_Level",
    "Traffic_Avg_Speed_KMH",
    # Weather factors
    "Weather_Visibility_KM",
    "Weather_Rainfall_mm",
    # Ride factors
    "Driver_Rating",
    "Ride_Surge_Pricing_Factor",
]

X = df[features].copy()

# ----------------------------------------------------------------------
# 3. PRE-PROCESSING
# ----------------------------------------------------------------------
# Missing values (Crime_Severity has 250 NaNs) -> median imputation
X = X.fillna(X.median())

# Scale: K-Means / DBSCAN are distance based, so every feature must have
# the same weight (mean 0, std 1)
X_scaled = StandardScaler().fit_transform(X)

# ----------------------------------------------------------------------
# 4. CHOOSE NUMBER OF CLUSTERS (Elbow + Silhouette)
# ----------------------------------------------------------------------
k_range = range(2, 11)
inertia, sil = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(k_range, inertia, "o-")
ax[0].set(title="Elbow Method", xlabel="Number of clusters (k)", ylabel="Inertia (WCSS)")
ax[1].plot(k_range, sil, "s-", color="darkorange")
ax[1].set(title="Silhouette Score", xlabel="Number of clusters (k)", ylabel="Score")
plt.tight_layout()
plt.savefig("1_elbow_silhouette.png", dpi=150)
plt.close()

best_k = 4   # chosen for interpretability; see elbow/silhouette plot
print("Silhouette by k:", dict(zip(k_range, np.round(sil, 3))))

# ----------------------------------------------------------------------
# 5. FIT THE ALGORITHMS
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
df["KMeans"] = kmeans.fit_predict(X_scaled)

hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
df["Hierarchical"] = hier.fit_predict(X_scaled)

gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE)
df["GMM"] = gmm.fit_predict(X_scaled)

dbscan = DBSCAN(eps=2.2, min_samples=15)
df["DBSCAN"] = dbscan.fit_predict(X_scaled)   # -1 = noise / outlier

# ----------------------------------------------------------------------
# 6. COMPARE ALGORITHMS
# ----------------------------------------------------------------------
rows = []
for name in ["KMeans", "Hierarchical", "GMM", "DBSCAN"]:
    labels = df[name].values
    mask = labels != -1                       # ignore DBSCAN noise
    n_clusters = len(set(labels[mask]))
    if n_clusters > 1:
        rows.append({
            "Algorithm": name,
            "Clusters": n_clusters,
            "Noise points": int((~mask).sum()),
            "Silhouette (higher better)": round(silhouette_score(X_scaled[mask], labels[mask]), 3),
            "Davies-Bouldin (lower better)": round(davies_bouldin_score(X_scaled[mask], labels[mask]), 3),
            "Calinski-Harabasz (higher better)": round(calinski_harabasz_score(X_scaled[mask], labels[mask]), 1),
        })
    else:
        rows.append({"Algorithm": name, "Clusters": n_clusters,
                     "Noise points": int((~mask).sum())})
comparison = pd.DataFrame(rows)
print("\n=== Algorithm comparison ===")
print(comparison.to_string(index=False))
comparison.to_csv("algorithm_comparison.csv", index=False)

# ----------------------------------------------------------------------
# 7. VISUALISE CLUSTERS (PCA -> 2-D)
# ----------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)
df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
var = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, name in zip(axes.ravel(), ["KMeans", "Hierarchical", "GMM", "DBSCAN"]):
    sns.scatterplot(data=df, x="PC1", y="PC2", hue=name, palette="Set2",
                    s=18, alpha=0.75, ax=ax, legend="full")
    ax.set_title(f"{name} clusters")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
plt.tight_layout()
plt.savefig("2_clusters_all_algorithms.png", dpi=150)
plt.close()

# Main K-Means picture with centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.figure(figsize=(9, 7))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="KMeans", palette="Set2",
                s=22, alpha=0.75, legend="full")
plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=250,
            edgecolor="white", linewidth=1.5, label="Centroids")
plt.title("K-Means Clusters of Ride Safety Data (PCA projection)")
plt.xlabel(f"PC1 ({var[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({var[1]:.1f}% variance)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("3_kmeans_clusters.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 8. PROFILE THE CLUSTERS  (what makes each cluster different?)
# ----------------------------------------------------------------------
profile = df.groupby("KMeans")[features].mean()
print("\n=== K-Means cluster profile (mean of each factor) ===")
print(profile.round(2).T.to_string())
print("\nCluster sizes:\n", df["KMeans"].value_counts().sort_index())
profile.round(3).to_csv("kmeans_cluster_profile.csv")

# Heatmap of z-scores: red = above overall average, blue = below
z_profile = (profile - X.mean()) / X.std()
plt.figure(figsize=(11, 6))
sns.heatmap(z_profile.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            cbar_kws={"label": "Std. dev. from overall mean"})
plt.title("What defines each cluster? (K-Means, standardised feature means)")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig("4_cluster_profile_heatmap.png", dpi=150)
plt.close()

# Which features separate the clusters most?  (spread of z-scores)
importance = z_profile.abs().max() - z_profile.abs().min()
importance = z_profile.std().sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importance.plot(kind="barh", color="steelblue")
plt.title("Features that separate the clusters most")
plt.xlabel("Std. dev. of cluster means (higher = more discriminating)")
plt.tight_layout()
plt.savefig("5_feature_importance.png", dpi=150)
plt.close()

# Categorical view (not used for clustering, only to describe clusters)
print("\n=== Crime type share per cluster (%) ===")
print((pd.crosstab(df["KMeans"], df["Crime_Type"], normalize="index") * 100).round(1))
print("\n=== City share per cluster (%) ===")
print((pd.crosstab(df["KMeans"], df["City"], normalize="index") * 100).round(1))

df.to_csv("ride_safety_with_clusters.csv", index=False)
print("\nDone. Images and CSVs saved.")
