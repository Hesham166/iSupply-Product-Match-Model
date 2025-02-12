import torch
from torch.nn import functional as F


def candidate_ranking(predictor, query, candidates, query_price, candidate_prices, sort=True, tau=10.0):
    """
    Rank candidate seller names by combining cosine similarity of text embeddings
    with a price similarity factor, utilizing an embedding cache for efficiency.

    Args:
        predictor: An instance of Predictor.
        query (str): The product name.
        candidates (list of str): Candidate product names.
        query_price (float): Price of the query product.
        candidate_prices (list of float): Prices for each candidate.
        sort (bool): Whether to sort the results in descending order.
        tau (float): Temperature parameter controlling the decay of the price similarity.
            A higher tau reduces the impact of price differences.
    
    Returns:
        List[Tuple[str, float]]: List of tuples (candidate, combined_score) sorted in descending order.
    """
    # Ensure the predictor has an embedding cache dictionary.
    if not hasattr(predictor, 'embedding_cache'):
        predictor.embedding_cache = {}

    with torch.no_grad():
        # ----- Query Embedding -----
        if query in predictor.embedding_cache:
            query_emb = predictor.embedding_cache[query]
        else:
            query_tensor = predictor.tokenize(query)  # Shape: (1, max_len)
            query_emb = predictor.model.get_embedding(query_tensor)  # Shape: (1, emb_dim)
            predictor.embedding_cache[query] = query_emb

        # ----- Candidate Embeddings with Caching -----
        # Preallocate a list to hold embeddings in the correct order.
        candidate_embs_list = [None] * len(candidates)
        non_cached_indices = []
        non_cached_candidates = []

        for i, candidate in enumerate(candidates):
            if candidate in predictor.embedding_cache:
                candidate_embs_list[i] = predictor.embedding_cache[candidate]
            else:
                non_cached_indices.append(i)
                non_cached_candidates.append(candidate)

        # Process all candidates not in the cache in one batch.
        if non_cached_candidates:
            candidate_tensors = torch.cat(
                [predictor.tokenize(candidate) for candidate in non_cached_candidates],
                dim=0
            )
            new_candidate_embs = predictor.model.get_embedding(candidate_tensors)  # Shape: (M, emb_dim)
            # Save each new embedding in the cache and in the list.
            for idx, emb in zip(non_cached_indices, new_candidate_embs):
                emb = emb.unsqueeze(0)  # Ensure shape is (1, emb_dim)
                candidate_embs_list[idx] = emb
                predictor.embedding_cache[candidates[idx]] = emb

        # Concatenate candidate embeddings into one tensor of shape (N, emb_dim)
        candidate_embs = torch.cat(candidate_embs_list, dim=0)

        # ----- Scoring -----
        # Compute cosine similarity between the query and each candidate.
        cosine_scores = F.cosine_similarity(query_emb.expand(candidate_embs.size(0), -1),
                                            candidate_embs, dim=1)

        # Compute the price similarity in a vectorized manner.
        candidate_prices_tensor = torch.tensor(candidate_prices, dtype=torch.float, device=predictor.device)
        price_similarity = torch.exp(-torch.abs(candidate_prices_tensor - query_price) / tau)

        combined_scores = cosine_scores * price_similarity

        # Pair each candidate with its score and sort if requested.
        candidate_scores = list(zip(candidates, combined_scores.tolist()))
        if sort:
            candidate_scores.sort(key=lambda x: x[1], reverse=True)

        return candidate_scores
