import os
import logging
import pickle
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

import config
from .dataset import ProductMatchingDataset
from .model import DeezyMatch
from .utils import setup_logging, save_checkpoint, build_vocab


def train_model(
    model, dataloader, criterion, optimizer, device, num_epochs=10, checkpoint_path=None
):
    """
    Trains the given model for a specified number of epochs.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader providing the training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device (torch.device): Device to run training on.
        num_epochs (int): Number of training epochs.
        checkpoint_path (str): Path to directory to save checkpoint files.

    Returns:
        nn.Module: The trained model.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x1, x2, price_diff, labels in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            price_diff = price_diff.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(x1, x2, price_diff)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save a checkpoint after each epoch if checkpoint_path is provided.
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(
                checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, filename=checkpoint_file)

    return model


import warnings
warnings.filterwarnings("ignore", message=".*'src.train' found in sys.modules.*")
if __name__ == "__main__":
    log_file = getattr(config, "LOG_FILE", None)
    setup_logging(log_file)

    data_csv_path = config.TRAIN_DATA
    df = pd.read_csv(data_csv_path)

    vocab = build_vocab(df, ["marketplace_product_name_ar", "seller_item_name"])
    logging.info(f"Vocabulary size: {len(vocab)}")

    vocab_path = getattr(config, 'VOCAB_PATH', 'checkpoints/vocab.pkl')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    logging.info(f"Saved vocabulary to {vocab_path}")

    logging.info("Preparing dataloader and model for training")
    max_len = getattr(config, "MAX_LEN", 50)
    dataset = ProductMatchingDataset(df, vocab, max_len=max_len, neg_ratio=getattr(config, "NEG_RATIO", 1))
    batch_size = getattr(config, "BATCH_SIZE", 64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,)

    device = config.DEVICE
    model = DeezyMatch(
        vocab_size=len(vocab),
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
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=getattr(config, 'LEARNING_RATE', 0.001))

    num_epochs = getattr(config, "NUM_EPOCHS", 10)
    checkpoint_dir = getattr(config, "CHECKPOINT_DIR", "checkpoints")

    logging.info('----------------------- Start Training -----------------------')
    trained_model = train_model(
        model, dataloader, criterion, optimizer, device,
            num_epochs=num_epochs,
            checkpoint_path=checkpoint_dir
        )
    logging.info('------------------------ End Training ------------------------')
    
