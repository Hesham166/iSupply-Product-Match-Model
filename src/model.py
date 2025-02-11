import torch
from torch import nn
from torch.nn import functional as F


class DeezyMatch(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, price_feature_dim=8, 
                 hidden_dim=64, num_layers=1, bidirectional=True, rnn_type='LSTM', 
                 combination='all', classifier_hidden_dim=32, dropout=0.2):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            price_feature_dim (int): Dimension for the price feature representation.
            hidden_dim (int): Hidden state dimension of the recurrent layer.
            num_layers (int): Number of RNN layers.
            bidirectional (bool): Use a bidirectional RNN.
            rnn_type (str): One of 'LSTM', 'GRU', or 'RNN'.
            combination (str): How to combine the two branch outputs.
                Options: 'concat', 'abs_diff', 'mul', or 'all'.
            classifier_hidden_dim (int): Hidden dimension for the classifier MLP.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.combination = combination.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=bidirectional,
                              nonlinearity='tanh', dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("Unsupported RNN type. Choose 'LSTM', 'GRU' or 'RNN'.")

        if self.combination == 'concat':
            combined_dim = 2 * self.output_dim
        elif self.combination in ['abs_diff', 'mul']:
            combined_dim = self.output_dim
        elif self.combination == 'all':
            combined_dim = 4 * self.output_dim
        else:
            raise ValueError("Unsupported combination method.")
        
        self.price_fc = nn.Linear(1, price_feature_dim)

        combined_dim = combined_dim + price_feature_dim

        self.fc_hidden = nn.Linear(combined_dim, classifier_hidden_dim)
        self.out = nn.Linear(classifier_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward_once(self, x):
        """
        Encode a single input sequence.
        Args:
            x (Tensor): Tensor of shape (batch, seq_len)
        Returns:
            Tensor: Encodes representation of shape (batch, output_dim)
        """
        emb = self.embedding(x)
        
        if self.rnn_type == 'lstm':
            _, (h_n, _) = self.rnn(emb)
        else:
            _, h_n = self.rnn(emb)

        if self.bidirectional:
            # h_n shape: (num_layer * 2, batch, hidden_dim); take the 
            # last layer's forward and backward.
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat((h_forward, h_backward), dim=1)
        else:
            h = h_n[-1, :, :]

        h = self.dropout(h)
        return h
    
    def forward(self, x1, x2, price_diff):
        """
        Processes a pair of text sequences along with a price difference.
        Args:
            x1, x2 (Tensor): Each of shape (batch, seq_len)
            price_diff (Tensor): Tensor of shape (batch, 1) with the absolute price difference.
        Returns:
            Tensor: Matching score with values in [0, 1]
        """
        v1 = self.forward_once(x1)
        v2 = self.forward_once(x2)

        if self.combination == 'concat':
            combined_text = torch.cat((v1, v2), dim=1)
        elif self.combination == 'abs_diff':
            combined_text = torch.abs(v1 - v2)
        elif self.combination == 'mul':
            combined_text = v1 * v2
        elif self.combination == 'all':
            combined_text = torch.cat((v1, v2, torch.abs(v1 - v2), v1 * v2), dim=1)
        else:
            raise ValueError("Unsupported combination method.")

        # Process the price difference.
        price_emb = F.relu(self.price_fc(price_diff))
        
        # Concatenate text and price features.
        combined = torch.cat((combined_text, price_emb), dim=1)
        hidden = F.relu(self.fc_hidden(combined))
        hidden = self.dropout(hidden)
        out = torch.sigmoid(self.out(hidden))
        return out.squeeze()

    def get_embedding(self, x):
        """
        Return the learned text embedding for a given input sequence (used in candidate ranking).
        Note: This does not include price information.
        """
        return self.forward_once(x)