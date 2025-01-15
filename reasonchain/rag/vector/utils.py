from reasonchain.utils. lazy_imports import os, numpy as np, sklearn


def pad_embedding(embedding, target_dim):
    """
    Pads an embedding to match the target dimension by adding zeros.
    Ensures consistent dimensions before concatenating.
    """
    # Ensure the embedding is a 2D array
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)  # Reshape to (1, embedding_dim)
    
    # Calculate the padding size
    current_dim = embedding.shape[-1]
    if current_dim < target_dim:
        padding = np.zeros((embedding.shape[0], target_dim - current_dim))  # Match batch size
        return np.hstack((embedding, padding))
    return embedding

def extract_name_from_path(db_path):
    """Extract the collection or index name from the database path."""
    return os.path.splitext(os.path.basename(db_path))[0]

# Reduce clip_embeddings to the same dimension as the database embeddings (384)
def resize_embeddings(clip_embeddings, target_dim=384):
    pca = sklearn.decomposition.PCA(n_components=target_dim)
    resized_embeddings = pca.fit_transform(clip_embeddings)
    return resized_embeddings