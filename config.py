# Use 'cuda' or 'cpu'
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tokenized sequence max length
MAX_LEN = 50

# Negative to positive samples ratio used in training
NEG_RATIO = 3

# Training data location
TRAIN_DATA = './data/preprocessed/training_dataset.csv'

# Model hyperparameters
MODEL_CONFIG = {
    "embedding_dim": 32,
    "price_feature_dim": 8,
    "hidden_dim": 64,
    "num_layers": 1,
    "bidirectional": True,
    "rnn_type": 'LSTM',
    "combination": 'all',
    "classifier_hidden_dim": 32,
    "dropout": 0.2
}

# Training 
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
CHECKPOINT_DIR = "checkpoints"

# Paths for prediction/training resources
CHECKPOINT_PATH = f'checkpoints/checkpoint_epoch_{NUM_EPOCHS}.pth'
VOCAB_PATH = 'checkpoints/vocab.pkl'

# Set log file
LOG_FILE = "logs/training.log"