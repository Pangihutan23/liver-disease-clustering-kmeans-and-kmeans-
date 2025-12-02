import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


df = pd.read_csv('hcvdat0.csv', sep=';')

print("Shape:", df.shape)
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

df['Category_raw'] = df['Category']
df = df.drop(columns=['Category'])

df['Sex_num'] = df['Sex'].map({'m':0, 'f':1})

numeric_cols = ['Sex_num','Age','ALB','ALP','ALT','AST',
                'BIL','CHE','CHOL','CREA','GGT','PROT']

X = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

from sklearn.metrics import silhouette_score

K_range = range(2, 10)
inertia = []
sil_scores = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    inertia.append(kmeans_temp.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_temp))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(K_range, inertia, '-o')
plt.title('Elbow Method (K-Means Standard)')
plt.xlabel('k')
plt.ylabel('Inertia')

plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, '-o')
plt.title('Silhouette Score (K-Means Standard)')
plt.xlabel('k')
plt.ylabel('Score')

plt.show()

inertia_kpp = []
sil_scores_kpp = []

for k in K_range:
    kmeans_kpp_temp = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    labels_kpp_temp = kmeans_kpp_temp.fit_predict(X_scaled)
    inertia_kpp.append(kmeans_kpp_temp.inertia_)
    sil_scores_kpp.append(silhouette_score(X_scaled, labels_kpp_temp))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(K_range, inertia_kpp, '-o')
plt.title('Elbow Method (K-Means++)')
plt.xlabel('k')
plt.ylabel('Inertia')

plt.subplot(1,2,2)
plt.plot(K_range, sil_scores_kpp, '-o')
plt.title('Silhouette Score (K-Means++)')
plt.xlabel('k')
plt.ylabel('Score')

plt.show()

k_final = 3

kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=50)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nJumlah anggota tiap cluster:")
print(df['Cluster'].value_counts())

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=X.columns)

print("\nCentroid (skala asli):")
print(centroids_df)

print("\nRata-rata fitur tiap cluster:")
print(df.groupby('Cluster')[numeric_cols].mean())

print("\nCross-tab Cluster vs Category:")
print(pd.crosstab(df['Cluster'], df['Category_raw']))

kmeans_plus = KMeans(
    n_clusters=k_final,
    init='k-means++',
    random_state=42,
    n_init=50
)

df['Cluster_kpp'] = kmeans_plus.fit_predict(X_scaled)

print("\n=================================================")
print("=== HASIL K-MEANS++ ===")
print("=================================================\n")

print("Jumlah anggota tiap cluster (K-Means++):")
print(df['Cluster_kpp'].value_counts())

centroids_kpp = scaler.inverse_transform(kmeans_plus.cluster_centers_)
centroids_kpp_df = pd.DataFrame(centroids_kpp, columns=X.columns)

print("\nCentroid K-Means++ (skala asli):")
print(centroids_kpp_df)

print("\nRata-rata fitur tiap cluster (K-Means++):")
print(df.groupby('Cluster_kpp')[numeric_cols].mean())

print("\nCross-tab Cluster K-Means++ vs Category:")
print(pd.crosstab(df['Cluster_kpp'], df['Category_raw']))


def compute_mse(X_scaled, model, labels):
    dist = []
    for i, x in enumerate(X_scaled):
        c = model.cluster_centers_[labels[i]]
        dist.append(np.sum((x - c)**2))
    return np.mean(dist)

mse_kmeans = compute_mse(X_scaled, kmeans, df['Cluster'])
mse_kpp = compute_mse(X_scaled, kmeans_plus, df['Cluster_kpp'])

print("\n===============================================")
print("🔍 EVALUASI K-MEANS STANDARD")
print("===============================================")
print(f"MSE (K-Means Standard)       : {mse_kmeans:.4f}")

print("\n===============================================")
print("🔍 EVALUASI K-MEANS++")
print("===============================================")
print(f"MSE (K-Means++)              : {mse_kpp:.4f}")
print("===============================================\n")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(pca_result[:,0], pca_result[:,1], c=df['Cluster'], cmap='tab10')
plt.title("PCA Visualization of Clusters (K-Means Standard)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Cluster')
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(pca_result[:,0], pca_result[:,1], c=df['Cluster_kpp'], cmap='tab10')
plt.title("PCA Visualization of Clusters (K-Means++)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Cluster')
plt.show()
