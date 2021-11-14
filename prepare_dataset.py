import json

from utils import create_vocabulary, download_dataset, extract_embeddings


if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    train_set, _, _ = download_dataset(config["dataset_dir"])
    vocab = create_vocabulary(train_set, config["vocab_size"])
    # Extract GloVe embeddings for tokens present in the training set vocab
    extract_embeddings(
        glove_dir=config["glove_dir"],
        save_path=config["embeddings"]["path"],
        vocab=vocab,
        embedding_dim=config["embeddings"]["size"]
    )
