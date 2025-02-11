import os
import pickle
import torch
from torch.nn import functional as F

import config
from .model import DeezyMatch
from .utils import tokenize
from .candidate_ranking import candidate_ranking


class Predictor:
    def __init__(self):
        """
        Initializes the predictor using the configuration file (config.py).
        """
        self.max_len = config.MAX_LEN
        self.device = config.DEVICE

        with open(config.VOCAB_PATH, 'rb') as f:
            self.vocab = pickle.load(f)

        self.model = DeezyMatch(
            len(self.vocab),
            embedding_dim=config.MODEL_CONFIG["embedding_dim"],
            price_feature_dim=config.MODEL_CONFIG["price_feature_dim"],
            hidden_dim=config.MODEL_CONFIG["hidden_dim"],
            num_layers=config.MODEL_CONFIG["num_layers"],
            bidirectional=config.MODEL_CONFIG["bidirectional"],
            rnn_type=config.MODEL_CONFIG["rnn_type"],
            combination=config.MODEL_CONFIG["combination"],
            classifier_hidden_dim=config.MODEL_CONFIG["classifier_hidden_dim"],
            dropout=config.MODEL_CONFIG["dropout"]
        )

        print(config.CHECKPOINT_PATH)

        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        if self.device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear, torch.nn.GRU, torch.nn.LSTM}, dtype=torch.qint8
            )

        self.model.eval()
        self._tokenize_cache = {}

    def tokenize(self, text):
        """
        Tokenize a given text.
        """
        if text in self._tokenize_cache:
            return self._tokenize_cache[text]
        tokenized = tokenize(text, self.vocab, self.max_len).unsqueeze(0).to(self.device)
        self._tokenize_cache[text] = tokenized
        return tokenized
    
    def candidate_ranking(self, query, candidates, query_price, candidate_prices, sort=True):
        """
        Rank candidate seller names by combining cosine similarity of text embeddings
        with a price similarity.

        Args:
            query (str): The product name.
            candidates (list of str): Candidate product names.
            query_price (float): Price of the query product.
            candidate_prices (list of float): Prices for each candidate.
            sort (bool): Whether to sort the results in descending order.

        Returns:
            List: List of tuples (candidate, combined_score, idx) sorted in descending order.
        """
        scores = candidate_ranking(self, query, candidates, query_price, candidate_prices, sort=False, tau=10)
        final_list = [(cand, score, i) for i, (cand, score) in enumerate(scores)]
        if sort:
            final_list = sorted(final_list, key=lambda x: x[1], reverse=True)
        return final_list