from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_emb(X):
    similarity = cosine_similarity(X, X)
    return similarity