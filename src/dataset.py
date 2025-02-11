import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import tokenize


class ProductMatchingDataset(Dataset):
    def __init__(self, df, vocab, max_len=50, neg_ratio=1):
        """
        Args:
            df (DataFrame): DataFrame with columns: sku, marketplace_product_name_ar, seller_item_name, price.
            vocab (dict): Mapping from characters to indices.
            max_len (int): Maximum sequence length (for padding/truncation).
            neg_ratio (int): Number of negative samples per positive sample.
        """
        self.df = df.copy()
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []

        self.df['price'] = self.df['price'].astype(float)

        # Create positive samples.
        positives = list(zip(
            self.df['marketplace_product_name_ar'].tolist(),
            self.df['seller_item_name'].tolist(),
            [0.0] * len(self.df),  # price difference is zero for positive pairs
            [1] * len(self.df)
        ))

        # Create negative samples.
        all_indices = np.arange(len(self.df))
        sku_to_indices = self.df.groupby('sku').indices  # dict: sku -> array of indices
        negative_indices_by_sku = {
            sku: np.setdiff1d(all_indices, indices)
            for sku, indices in sku_to_indices.items()
        }

        negatives = []
        for row in self.df.itertuples(index=True):
            current_sku = row.sku
            neg_indices = negative_indices_by_sku.get(current_sku)
            if neg_indices.size > 0:
                for _ in range(neg_ratio):
                    rand_idx = np.random.choice(neg_indices)
                    neg_row = self.df.iloc[rand_idx]
                    price_diff = abs(row.price - neg_row['price'])
                    negatives.append((
                        row.marketplace_product_name_ar,
                        neg_row['seller_item_name'],
                        price_diff,
                        0
                    ))
        
        # Combine positive and negative samples.
        self.samples = positives + negatives

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        marketplace, seller, price_diff, label = self.samples[idx]
        tokens_marketplace = tokenize(marketplace, self.vocab, self.max_len)
        tokens_seller = tokenize(seller, self.vocab, self.max_len)
        price_tensor = torch.tensor([price_diff], dtype=torch.float)
        return tokens_marketplace, tokens_seller, price_tensor, torch.tensor(label, dtype=torch.float)
