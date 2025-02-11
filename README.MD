# Product Matching Deep Learning Pipeline Documentation

This project implements a deep learning pipeline for product matching. It is designed to compare product names and seller item names—and to incorporate price 
differences—to determine if two products match. The project uses a recurrent neural network (RNN) architecture (with options for LSTM, GRU, or vanilla RNN) 
to encode text at the character level, and then combines the resulting representations with a price similarity factor to produce a final matching score. The 
project allows for a high level of flexibity and different configurations.

> **_Note:_** The model architecture is inspired by [DeezyMatch: A Flexible Deep Learning Approach to Fuzzy String Matching](https://aclanthology.org/2020.emnlp-demos.9/) paper.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Module-by-Module Description](#module-by-module-description)
    - config.py
    - model.py
    - dataset.py
    - candidate_ranking.py
    - utils.py
    - train.py
    - predictor.py
4. [Setup and Usage](#setup-and-usage)

---

## Overview

The pipeline is composed of two main stages:

- **Training:**  
  A training script prepares data, builds a character-level vocabulary from text columns, constructs a dataset with both positive and negative samples, and trains the `DeezyMatch` model. The training process uses binary cross-entropy loss and saves checkpoints after each epoch.

- **Prediction / Inference:**  
  A `Predictor` class loads a trained model from a checkpoint and provides methods to tokenize text inputs and rank candidate product names. The candidate ranking is based on a combination of the cosine similarity between text embeddings and an exponential decay function over the price differences.

The project relies on PyTorch for deep learning, and uses Pandas and NumPy for data handling.

---

## File Structure

```
iSUPPLY
├── config.py                       # Global configuration parameters.
├── src                             
|   ├── model.py                    # Definition of the DeezyMatch model.
|   ├── dataset.py                  # Custom Dataset class for product matching.
|   ├── candidate_ranking.py        # Function to rank candidate products.
|   ├── utils.py                    # Utility functions (logging, tokenization, data cleaning, etc.).
|   ├── train.py                    # Training script that builds vocab, trains the model, and saves checkpoints.
|   └── predictor.py                # Predictor class for loading the trained model and performing inference.
├── logs                            
|   ├── training.log                # Contains the training epoch losses.
├── data
|   ├── raw                         # Contains xlsx file before preprocessing.
|   ├── preprocessed                # Contains preprocessed training, validation, and master file.
├── checkpoints                     # Contains training vocab.pkl and training checkpoints created after every epoch.
├── proprocessing.ipynb             # A notebook that demonstrates how to preprocess xlsx file to csv file.
|── test.ipynb                      # A notebook that shows how to use the model.
└── validation_output.xlsx          # The output of the model after using it on validation dataset.

```

---

## Modules Description

- **config.py**  
  Global settings including device selection, hyperparameters, data paths, and training parameters.

- **model.py**  
  Defines the `DeezyMatch` model that:
  - Embeds text using an embedding layer.
  - Encodes sequences with an RNN.
  - Combines two encoded sequences using methods such as concatenation, absolute difference, multiplication, or all.
  - Integrates price features via a linear layer and outputs a matching score.

- **dataset.py**  
  Implements `ProductMatchingDataset`, which creates:
  - **Positive samples**: Pairs from the same product (sku) with a price difference of 0.
  - **Negative samples**: Pairs from different products with calculated price differences.
  Uses character-level tokenization and fixed-length padding.

- **candidate_ranking.py**  
  Provides a function to rank candidate seller names by:
  - Calculating cosine similarity between query and candidate embeddings.
  - Adjusting scores with an exponential decay function based on price differences.
  - Returning a sorted list of candidates with their combined scores.

- **utils.py**  
  Contains utility functions for:
  - Logging and checkpoint save/load.
  - Building a character-level vocabulary.
  - Tokenization and text cleaning.
  - Saving DataFrames to CSV/Excel.

  > This module provides many helper function that could make the process of using the model smooth, you can see that in `test.ipynb` and `preprocess_data.ipynb`.

- **train.py**  
  Orchestrates model training by:
  - Loading the preprocessed training data.
  - Building the vocabulary.
  - Creating training samples (both positive and negative).
  - Training the model using binary cross-entropy loss.
  - Saving checkpoints after each epoch.

- **predictor.py**  
  Provides the `Predictor` class for inference:
  - Loads the saved vocabulary and model checkpoint.
  - Applies dynamic quantization for CPU inference.
  - Caches tokenized texts.
  - Offers a candidate ranking method based on text and price similarities.

---

## Setup and Usage

### Environment Setup

- **Dependencies:**  
  - Python 3.x  
  - PyTorch  
  - Pandas  
  - NumPy  
  - (Optionally) openpyxl (for saving Excel files)  
  - Any other libraries required (e.g., logging, pickle, os, re, etc.)

### Training the Model

1. **Prepare Your Data:**  
   - Use `utils.excel_to_csv_pipeline()` to turn you excell file to csv file (like in `preprocess_data.ipynb`).
   - Ensure that the training data CSV is available at the path specified by `config.TRAIN_DATA`.
   - The CSV should contain the following columns:  
     - `sku`  
     - `marketplace_product_name_ar`  
     - `seller_item_name`  
     - `price`

2. **Run the Training Script:**

    While in the project root directory, run the following command:
   ```bash
   python -m src.train
   ```
   - This will:
     - Set up logging.
     - Build the character-level vocabulary.
     - Create positive and negative samples.
     - Train the `DeezyMatch` model for the number of epochs specified.
     - Save checkpoints after each epoch in the `checkpoints` directory.
     - Save the vocabulary to `checkpoints/vocab.pkl`.

### Performing Inference

> **_Note:_** You must be in the project root directory for this to work (see `test.ipynb`).

1. **Load the Predictor:**  
   In your inference script or interactive session, import and instantiate the Predictor class:
   ```python
   from src.predictor import Predictor

   predictor = Predictor()
   ```

2. **Rank Candidate Products:**  
   Use the `candidate_ranking` method to rank candidates given a query product name and associated prices:
   ```python
   query = "Example Product Name"
   candidates = ["Candidate 1", "Candidate 2", "Candidate 3"]
   query_price = 100.0
   candidate_prices = [95.0, 105.0, 110.0]

   ranked_candidates = predictor.candidate_ranking(query, candidates, query_price, candidate_prices)
   for candidate, score, idx in ranked_candidates:
       print(f"Candidate: {candidate}, Score: {score:.4f}, Index: {idx}")
   ```