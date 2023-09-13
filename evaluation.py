import numpy as np

def recall_at_k(ranked_keywords, pos_keywords, k):
    return len(set(ranked_keywords) & set(pos_keywords)) / len(pos_keywords)

def precision_at_k(ranked_keywords, pos_keywords, k):
    return len(set(ranked_keywords[:k]) & set(pos_keywords)) / k

def NDCG_at_k(ranked_keywords, pos_keywords, k):
    dcg = 0
    for i, keyword in enumerate(ranked_keywords[:k]):
        if keyword in pos_keywords:
            dcg += 1 / np.log2(i+2)
    idcg = sum([1 / np.log2(i+2) for i in range(min(k, len(pos_keywords)))])
    ndcg = dcg / idcg
    return ndcg

def mean_reciprocal_rank_at_k(ranked_keywords, pos_keywords):
    for i, keyword in enumerate(ranked_keywords):
        if keyword in pos_keywords:
            return 1 / (i+1)
    return 0

def hit_rate_at_k(ranked_keywords, pos_keywords, k):
    return 1 if len(set(ranked_keywords[:k]) & set(pos_keywords)) > 0 else 0

def F1_at_k(ranked_keywords, pos_keywords, k):
    precision = precision_at_k(ranked_keywords, pos_keywords, k)
    recall = recall_at_k(ranked_keywords, pos_keywords, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)