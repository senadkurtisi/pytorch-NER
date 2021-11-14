import os
import json
import shutil
from collections import Counter

import torch
import numpy as np
from datasets import load_dataset


def log_gradient_norm(model, writer, step, mode, norm_type=2):
    """Writes model param's gradients norm to tensorboard"""
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    writer.add_scalar(f"Gradient/{mode}", total_norm, step)


def save_checkpoint(model, start_time, epoch):
    """Saves specified model checkpoint."""
    target_dir = f"checkpoints\\{start_time}"
    os.makedirs(target_dir, exist_ok=True)
    # Save model weights
    save_path = f"{target_dir}\\model_{epoch}.pth"
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)

    # Save model configuration
    if not os.path.exists(f"{target_dir}\\config.json"):
        shutil.copy("config.json", os.path.join(target_dir, "config.json"))
        shutil.copy("classifier.py", os.path.join(target_dir, "classifier.py"))
        shutil.copy("transformer.py", os.path.join(target_dir, "transformer.py"))
        shutil.copy("utils.py", os.path.join(target_dir, "utils.py"))


def process_subset(subset, separator="\t"):
    """Processes the given subset.

    Extracts the input tokens (words) and labels for each sequence in the subset.
    Forms a representation string for each sample in the following format:
        SEQ_LEN [SEPARATOR] INPUT_TOKENS [SEPARATOR] LABELS
        where:
            SEQ_LEN - Number of tokens in that particular sequence
            INPUT_TOKENS - Input tokens separated by the @separator
            LABELS - Integer labels separated by the @separator
    """
    processed_subset = []
    max_len = 0
    for sample in subset:
        # Load and process tokens
        tokens = sample["tokens"]
        tokens = [token.strip() for token in tokens]
        # Load and process NER tags
        ner_tags = sample["ner_tags"]
        ner_tags = [str(tag) for tag in ner_tags]

        sample_size = len(sample["tokens"])
        max_len = max(max_len, sample_size)

        processed_sample = f"{sample_size}{separator}"
        processed_sample += separator.join(tokens + ner_tags) + "\n"
        processed_subset.append(processed_sample)

    return processed_subset


def save_subset(subset, dataset_dir, subset_name):
    """Saves processed subset to the desired dataset directory.

    Arguments:
        subset: Subset to save
        dataset_dir (str): Dataset directory to which to save subset
        subset_name (str): Name of the subset
    """
    if subset_name not in ["train", "validation", "test"]:
        raise ValueError(
            "Subset name invalid! Expected: train, validation or test but received {}".format(subset_name)
        )

    save_path = os.path.join(dataset_dir, "{}.txt".format(subset_name))
    with open(save_path, "w") as f:
        f.writelines(subset)


def download_dataset(dataset_dir):
    """Downloads the CoNLL2003 dataset from the HuggingFace
       and saves separate subsets into dataset directory.

    Arguments:
        dataset_dir (str): Directory to which to save the dataset subsets
    """
    # Download the dataset from the HuggingFace
    DATASET_NAME = "conll2003"
    dataset_group = load_dataset(DATASET_NAME)

    os.makedirs(dataset_dir, exist_ok=True)

    # Extract subsets of data
    train_set = dataset_group["train"]
    valid_set = dataset_group["validation"]
    test_set = dataset_group["test"]

    # Perform subset processing
    train_set_processed = process_subset(train_set)
    valid_set_processed = process_subset(valid_set)
    test_set_processed = process_subset(test_set)

    # Save processed subsets
    save_subset(train_set_processed, dataset_dir, "train")
    save_subset(valid_set_processed, dataset_dir, "validation")
    save_subset(test_set_processed, dataset_dir, "test")
    print("\nDataset downloaded and processed.")

    return train_set, valid_set, test_set


def create_vocabulary(train_set, vocab_size):
    """Creates vocabulary out of the training set tokens.

    Arguments:
        train_set: CoNLL2003 train_set from HuggingFace
        vocab_size (int): Maximum number of tokens in the vocab
    Returns:
        vocab (dict): Vocabulary of all tokens in the training set
            key: token
            value: ordinal number of token in the vocabulary
    """
    all_tokens = []
    for token_subseq in train_set["tokens"]:
        all_tokens += token_subseq

    # Perform some pre-processing of the tokens
    all_tokens_lower = list(map(str.lower, all_tokens))
    all_tokens_strip = list(map(str.strip, all_tokens_lower))

    # Count the occurence of every word
    counter = Counter(all_tokens_strip)
    # Extract VOCAB_SIZE - 2 since we will define tokens for padding elements
    # and words which aren't present in the training set
    most_frequent = counter.most_common(vocab_size - 2)

    # Initialize the vocabulary
    vocab = {
        "UNK": 0,
        "PADD": 1
    }
    ind = len(vocab)
    # Populate the vocab
    for token, _ in most_frequent:
        vocab[token] = ind
        ind += 1

    print("\nCreated vocabulary of {} tokens.".format(ind))
    return vocab


def extract_embeddings(config, vocab):
    """Extracts GloVe word embeddings for words in vocab.

    Arguments:
        config (object): Contains dataset & pipeline configuration info
        vocab (dict): word - ordinal number mapping
    """
    embeddings_config = config["embeddings"]
    save_path_emb = embeddings_config["path"]
    embedding_dim = embeddings_config["size"]

    save_path_map = config["word2idx_path"]
    # Used for finding the embedding vector for each token
    word_to_idx = {"<unk>": 0, "<pad>": 1}
    vectors = []

    idx = 0
    vocab_bias = len(word_to_idx)
    embedding_file_name = "glove.6B.{}d.txt".format(embedding_dim)
    embeddings_path = os.path.join(config["glove_dir"], embedding_file_name)
    with open(embeddings_path, "rb") as f:
        for line in f:
            line = line.decode().split()
            # Extract and pre-process the token
            word = line[0]
            word = word.strip().lower()
            # Remember the embedding vector if the word is in the vocab
            if word in vocab.keys():
                word_to_idx[word] = idx + vocab_bias
                embedding_vec = np.array(line[1:], dtype="float")
                vectors += [embedding_vec]
                idx += 1

    vectors = np.array(vectors)
    # Embedding vector for tokens used for padding the input sequence
    pad_embedding = np.zeros((embedding_dim,))
    # Embedding vector for tokens not present in the training set
    unk_embedding = vectors.mean(axis=0)

    vectors = np.vstack([unk_embedding, pad_embedding, vectors])
    # Save extracted embeddings
    np.savetxt(save_path_emb, vectors)
    # Save token:index mapping
    with open(save_path_map, "w", encoding="utf8") as f:
        json.dump(word_to_idx, f)

    print("\nExtracted GloVe embeddings for all tokens in the training set.") 
    print("Number of tokens:", vectors.shape[0], "Embedding vectors size:", embedding_dim)
