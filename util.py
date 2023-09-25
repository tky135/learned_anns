import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
def pca_sampling(X, n_samples, n_components=None, random_seed=None):
    """
    Generate samples based on PCA.

    Parameters:
    - X: Original data (numpy array of shape [n_samples_original, n_features])
    - n_samples: Number of samples to generate
    - n_components: Number of principal components to use (if None, it will be set to min(n_samples_original, n_features))
    - random_seed: Seed for reproducibility

    Returns:
    - Generated samples (numpy array of shape [n_samples, n_features])
    """
    # If n_components is not provided, set it to min(n_samples_original, n_features)
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    n_components = min(n_components, X.shape[0], X.shape[1])
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Determine the range of values in the PCA space
    min_vals = X_pca.min(axis=0)
    max_vals = X_pca.max(axis=0)
    
    # Generate random points in the PCA space
    np.random.seed(random_seed)  # Set seed for reproducibility
    random_pca_points = np.random.uniform(min_vals, max_vals, size=(n_samples, n_components))
    
    # Transform these points back to the original space
    generated_samples = pca.inverse_transform(random_pca_points)
    
    # Visualization
    # plt.figure(figsize=(10, 8))
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', alpha=0.6, label='Original Vectors')
    # plt.scatter(random_pca_points[:, 0], random_pca_points[:, 1], color='red', alpha=0.6, label='Generated Vectors')
    # plt.title('PCA-based Sampling Visualization')
    # plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]*100:.2f}%)')
    # plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]*100:.2f}%)')
    # plt.legend()
    # plt.savefig("figure.png")
    
    return generated_samples.astype(np.float32)

def generate_query_data(X, k, n_query=1000, random_seed=None):
    # sample query data from X's space
    np.random.seed(random_seed)
    query = pca_sampling(X, n_query, n_components=64, random_seed=random_seed)

    # find the nearest neighbor of each query
    q_nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = q_nbrs.kneighbors(query)
    indices = indices.reshape(-1)
    return query.astype(np.float32), indices.astype(np.int64)    
  
def fvecs_read(filename):
    """
    Returns a np.float32 array of N x D
    """

    with open(filename, 'rb') as f:
        # Read the vector size
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        vecsizeof = 4 + d * 4
        # Get the number of vectors
        f.seek(0, 2)
        num = f.tell() // vecsizeof
        f.seek(0, 0)
        # Read n vectors
        v = np.fromfile(f, dtype=np.float32, count=(d + 1) * num)
        v = v.reshape((num, d + 1))
        # Check if the first column (dimension of the vectors) is correct
        assert np.all(v[1:, 0] == v[0, 0])
        return v[:, 1:]
def ivecs_read(filename):
    """
    Returns an np.int32 array of N x D
    """
    with open(filename, 'rb') as f:
        # Read the vector size
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        vecsizeof = 4 + d * 4
        # Get the number of vectors
        f.seek(0, 2)
        num = f.tell() // vecsizeof
        f.seek(0, 0)
        # Read n vectors
        v = np.fromfile(f, dtype=np.int32, count=(d + 1) * num)
        v = v.reshape((num, d + 1))
        # Check if the first column (dimension of the vectors) is correct
        assert np.all(v[1:, 0] == v[0, 0])
        return v[:, 1:]

def generate_query_data_v1(X, n_clusters=10, n_query=1000, random_seed=None):
    
    # do a k-means clustering on X
    kmeans = KMeans(n_init=10, n_clusters=n_clusters).fit(X)
    kmeans = kmeans.fit(X)

    clusters = kmeans.predict(X)
    

    # sample query data from X's space
    np.random.seed(random_seed)
    query = pca_sampling(X, n_query, n_components=64, random_seed=random_seed)

    # find the nearest neighbor of each query
    q_nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X)
    distances, indices = q_nbrs.kneighbors(query)
    indices = indices.reshape(-1)

    nn_cluster_idx = clusters[indices]
    q_cluster_idx = kmeans.predict(query)
    print(q_cluster_idx.shape, nn_cluster_idx.shape)
    # for i in range(1000):
    #     print(q_cluster_idx[i], nn_cluster_idx[i])
    # plot the distribution of the query and its nearest neighbor
    # do some analysis here
    print("Query cluster vs nn cluster")
    print(np.mean(q_cluster_idx == nn_cluster_idx))

    temp_plot(query, X, -1, q_cluster_idx, nn_cluster_idx, clusters)
    return query.astype(np.float32), q_cluster_idx.astype(np.int64), nn_cluster_idx.astype(np.int64)
def temp_plot(query, X, query_id, q_cluster_idx, nn_cluster_idx, X_clusters):
    # Visualization
    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    random_pca_points = pca.transform(query)
    # print(q_cluster_idx[query_id])
    # print(nn_cluster_idx[query_id])
    plt.figure(figsize=(20, 16))
    for i in range(10):
        mask = X_clusters == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, label=f'Cluster {i}')

    # plt.scatter(X_pca[query_id, 0], X_pca[query_id, 1], color='blue', alpha=0.6, label='Original Vectors')
    plt.scatter(random_pca_points[query_id, 0], random_pca_points[query_id, 1], color='red', alpha=0.6, label='Generated Vectors', marker='x')
    plt.title('PCA-based Sampling Visualization')
    plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.legend()
    plt.savefig("figure.png")

if __name__ == "__main__":
    v = fvecs_read("data/siftsmall/siftsmall_base.fvecs")
    print(v.shape)
    v = v[:20]
    v.dump("vec_1000.npy")
    # generate_query_data(v, k=10, n_query=1000, random_seed=0)
    generate_query_data_v1(v, n_clusters=10, n_query=1000, random_seed=0)
    # i = ivecs_read("data/siftsmall/siftsmall_groundtruth.ivecs")