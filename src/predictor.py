import pickle
import torch
import pandas as pd

import config
from .model import DeezyMatch
from .utils import tokenize, add_top_k_scores_to_df, clean_df, save_dataframe_to_xlsx
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
    
    def predict_from_excel(
        self,
        master_xlsx_path,
        test_xlsx_path,
        output_xlsx_path,
        master_sheet="Sheet1",
        test_sheet="Sheet1",
        output_sheet_name="Sheet1",
        query_names_column=None,
        query_prices_column=None,
        master_candidate_names_column=None,
        master_prices_column=None,
        k=3,
    ):
        """
        Loads a master and test Excel file, preprocesses them, computes predictions,
        and outputs an Excel file with the top-k predictions added to the test data.

        Args:
            master_xlsx_path (str): Path to the master Excel file. This file should contain at least:
                - A candidate names column (provided via master_candidate_names_column)
                - A candidate prices column (provided via master_prices_column)
                - A 'sku' column for identifying candidates.
            test_xlsx_path (str): Path to the test Excel file. This file should contain:
                - A query names column (provided via query_names_column)
                - A query prices column (provided via query_prices_column)
            output_xlsx_path (str): Path (including filename) where the output Excel file will be saved.
            master_sheet (str): Sheet name in the master Excel file (default: "Sheet1").
            test_sheet (str): Sheet name in the test Excel file (default: "Sheet1").
            output_sheet_name (str): Sheet name in the output Excell file (default: "Sheet1").
            query_names_column (str): Column name in the test file with the query (product) names.
            query_prices_column (str): Column name in the test file with the query prices.
            master_candidate_names_column (str): Column name in the master file with candidate names.
            master_prices_column (str): Column name in the master file with candidate prices.
            k (int): Number of top predictions to add for each test record (default: 3).

        Returns:
            pd.DataFrame: The test DataFrame with additional columns for the top-k predictions.
        """
        # Load Excel files for master and test data
        master_df = pd.read_excel(master_xlsx_path, sheet_name=master_sheet)
        test_df = pd.read_excel(test_xlsx_path, sheet_name=test_sheet)
        print("Loaded master and test data from Excel files.")

        # Preprocess the DataFrames:
        # For the master data, clean the candidate names and price columns.
        master_df_clean = clean_df(
            master_df,
            text_columns=[master_candidate_names_column],
            price_column=master_prices_column,
            drop_duplicates=True,
        )
        # For the test data, clean the query names and price columns.
        test_df_clean = clean_df(
            test_df,
            text_columns=[query_names_column],
            price_column=query_prices_column,
            drop_duplicates=False,
        )
        print("Data cleaning completed.")

        # Prepare the candidate lists from the master file
        candidates = master_df_clean[master_candidate_names_column].tolist()
        candidate_prices = master_df_clean[master_prices_column].tolist()

        # For each test record, compute candidate ranking using the model
        print("Model start...")
        all_scores = []
        for idx, row in test_df_clean.iterrows():
            query = row[query_names_column]
            query_price = row[query_prices_column]
            ranking = self.candidate_ranking(
                query, candidates, query_price, candidate_prices, sort=True
            )
            all_scores.append(ranking)
        print("Model done finding most similar candidates.")

        # Append top-k predictions to the test DataFrame.
        # This function adds new columns (e.g. sku1, pred1, sim1, etc.) based on the ranking scores.
        test_df_with_preds = add_top_k_scores_to_df(test_df, all_scores, master_df_clean, k=k)

        # Save the updated test DataFrame to an Excel file
        save_dataframe_to_xlsx(test_df_with_preds, output_xlsx_path, output_sheet_name)
        print(f"Predictions saved to {output_xlsx_path}")

        return test_df_with_preds
