import torch
import torch.nn.functional as F
import math


def candidate_ranking(predictor, query, candidates, query_price, candidate_prices, sort=True, tau=10.0):
    """
    Rank candidate seller names by combining cosine similarity of text embeddings
    with a price similarity factor.

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
        List: List of tuples (candidate, combined_score) sorted in descending order.
    """
    with torch.no_grad():
        # Tokenize and compute the query embedding
        query_tensor = predictor.tokenize(query)             # Shape: (1, max_len)
        query_emb = predictor.model.get_embedding(query_tensor)  # Shape: (1, emb_dim)
        
        # Batch process candidate tokenizations:
        candidate_tensors = torch.cat([predictor.tokenize(candidate) for candidate in candidates], dim=0)
        candidate_embs = predictor.model.get_embedding(candidate_tensors)  # Shape: (N, emb_dim)
        
        # Expand query embedding to match the number of candidates and compute cosine similarities
        expanded_query_emb = query_emb.expand(candidate_embs.size(0), -1)
        cosine_scores = F.cosine_similarity(expanded_query_emb, candidate_embs, dim=1)

        # Compute a price similarity score for each candidate.
        # Here we use an exponential decay: higher price differences yield lower scores.
        price_similarity = torch.tensor(
            [math.exp(-abs(query_price - float(cp)) / tau) for cp in candidate_prices],
            dtype=torch.float,
            device=predictor.device
        )  # shape: (N,)

        combined_scores = cosine_scores * price_similarity
        
        # Zip candidates with their combined scores.
        candidate_scores = list(zip(candidates, combined_scores.tolist()))
        if sort:
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores