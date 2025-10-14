import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

"""Visualization helpers for small demo (avoid large memory use)."""

def tsne_plot(df, feature_col='features', label_col='name', title='t-SNE of OpenL3 Embeddings', perplexity=15):
    features = np.array(df[feature_col].tolist())
    labels = df[label_col].values
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(df)-1))
    emb2d = tsne.fit_transform(features)
    plt.figure(figsize=(8,5))
    for reciter in set(labels):
        idx = labels == reciter
        plt.scatter(emb2d[idx,0], emb2d[idx,1], label=reciter, alpha=0.7)
    plt.title(title)
    plt.legend(fontsize='small')
    plt.tight_layout()
    return emb2d

def similarity_heatmap(df, feature_col='features', label_col='name', title='Reciter Similarity Matrix'):
    reciter_avg = df.groupby(label_col)[feature_col].apply(lambda x: np.mean(np.vstack(x), axis=0))
    sim = cosine_similarity(list(reciter_avg))
    plt.figure(figsize=(8,6))
    sns.heatmap(sim, annot=True, fmt='.2f', xticklabels=reciter_avg.index, yticklabels=reciter_avg.index, cmap='YlGnBu')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return sim
