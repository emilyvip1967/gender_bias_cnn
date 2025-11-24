#!/usr/bin/env python
# cnn_gap_baseline.py
#
# CNN baseline for GAP coreference resolution using GloVe embeddings.
# Pairwise setup: (pronoun, candidate) -> coreference (0/1).

import os
import re
import math
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# Utils: seeds and device
# ---------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------

def load_gap_tsv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GAP file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    return df


def infer_pronoun_gender(pronoun: str) -> str:
    """Rough heuristic: map pronoun string to gender."""
    p = pronoun.strip().lower()
    male = {"he", "him", "his"}
    female = {"she", "her", "hers"}
    if p in male:
        return "M"
    if p in female:
        return "F"
    return "U"  # unknown/other


def build_pairwise_examples(df: pd.DataFrame):
    """
    For each GAP row, create two examples:
      - (Text, Pronoun, A) with label = A-coref
      - (Text, Pronoun, B) with label = B-coref
    We also keep the pronoun gender for bias analysis.
    """
    examples = []
    for _, row in df.iterrows():
        text = str(row["Text"])
        pronoun = str(row["Pronoun"])
        cand_a = str(row["A"])
        cand_b = str(row["B"])
        label_a = int(row["A-coref"])
        label_b = int(row["B-coref"])

        gender = infer_pronoun_gender(pronoun)

        # Simple template; you can tweak later if you want
        text_a = f"[PRON] {pronoun} [CAND] {cand_a} [CTX] {text}"
        text_b = f"[PRON] {pronoun} [CAND] {cand_b} [CTX] {text}"

        examples.append({"text": text_a, "label": label_a, "gender": gender})
        examples.append({"text": text_b, "label": label_b, "gender": gender})

    return examples


# ---------------------------------------------------------
# Tokenization & vocab
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\[pron\]|\[cand\]|\[ctx\]|\w+|\S")


def simple_tokenize(text: str):
    text = text.lower()
    return TOKEN_PATTERN.findall(text)


def build_vocab(examples, min_freq: int = 3, max_size: int = 30000):
    counter = Counter()
    for ex in examples:
        tokens = simple_tokenize(ex["text"])
        counter.update(tokens)

    stoi = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(stoi) >= max_size:
            break
        stoi[token] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


# ---------------------------------------------------------
# GloVe loading
# ---------------------------------------------------------

def load_glove_embeddings(stoi, glove_path: str, emb_dim: int = 300):
    if not os.path.isfile(glove_path):
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    embeddings = np.random.normal(scale=0.1,
                                  size=(len(stoi), emb_dim)).astype(np.float32)
    embeddings[0] = np.zeros(emb_dim, dtype=np.float32)  # <pad> zero

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < emb_dim + 1:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if word in stoi:
                idx = stoi[word]
                embeddings[idx] = vec
                found += 1

    print(f"[GloVe] Loaded vectors for {found}/{len(stoi)} tokens")
    return torch.tensor(embeddings)


# ---------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------

class GAPCNNDataset(Dataset):
    def __init__(self, examples, stoi, max_len: int = 128):
        self.examples = examples
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def _text_to_ids(self, text: str):
        tokens = simple_tokenize(text)
        ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        else:
            ids = ids + [self.stoi["<pad>"]] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text_ids = self._text_to_ids(ex["text"])
        label = ex["label"]
        gender = 1 if ex["gender"] == "M" else 0  # for bias stats if needed

        return (
            torch.tensor(text_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(gender, dtype=torch.long),
        )


# ---------------------------------------------------------
# CNN model
# ---------------------------------------------------------

class TextCNN(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_filters: int = 100,
        filter_sizes=(3, 4, 5),
        dropout: float = 0.5,
    ):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.size()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)       # (batch, seq_len, emb_dim)
        emb = emb.transpose(1, 2)            # (batch, emb_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            x = conv(emb)                    # (batch, num_filters, L')
            x = torch.relu(x)
            x = torch.max(x, dim=2).values   # (batch, num_filters)
            conv_outs.append(x)

        x = torch.cat(conv_outs, dim=1)      # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1)      # (batch,)
        return logits


# ---------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels, _genders in loader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += (preds == labels.long()).sum().item()
        total_count += input_ids.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def eval_model(model, loader, criterion, compute_gender_bias: bool = True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    male_correct = 0
    male_total = 0
    female_correct = 0
    female_total = 0

    with torch.no_grad():
        for input_ids, labels, genders in loader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            genders = genders.to(DEVICE)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_correct += (preds == labels.long()).sum().item()
            total_count += input_ids.size(0)

            if compute_gender_bias:
                # genders: 1 = male, 0 = non-male (female/unknown)
                male_mask = genders == 1
                female_mask = genders == 0

                if male_mask.any():
                    male_correct += (preds[male_mask] == labels.long()[male_mask]).sum().item()
                    male_total += male_mask.sum().item()

                if female_mask.any():
                    female_correct += (preds[female_mask] == labels.long()[female_mask]).sum().item()
                    female_total += female_mask.sum().item()

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count

    gender_stats = None
    if compute_gender_bias:
        male_acc = male_correct / male_total if male_total > 0 else math.nan
        female_acc = female_correct / female_total if female_total > 0 else math.nan
        gender_bias = male_acc - female_acc if (male_total > 0 and female_total > 0) else math.nan
        gender_stats = {
            "male_acc": male_acc,
            "male_total": male_total,
            "female_acc": female_acc,
            "female_total": female_total,
            "bias": gender_bias,
        }

    return avg_loss, avg_acc, gender_stats


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CNN baseline for GAP coreference.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing GAP tsv files.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="gap-development.tsv",
        help="Training TSV file name.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="gap-validation.tsv",
        help="Validation TSV file name.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="gap-test.tsv",
        help="Test TSV file name.",
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        required=True,
        help="Path to GloVe embedding file (e.g. glove.6B.300d.txt).",
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    print("Using device:", DEVICE)
    print("Arguments:", args)

    # Load GAP splits
    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, args.val_file)
    test_path = os.path.join(args.data_dir, args.test_file)

    print("\n====================================================")
    print("Loading GAP dataset...")
    train_df = load_gap_tsv(train_path)
    val_df = load_gap_tsv(val_path)
    test_df = load_gap_tsv(test_path)

    train_examples = build_pairwise_examples(train_df)
    val_examples = build_pairwise_examples(val_df)
    test_examples = build_pairwise_examples(test_df)

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")
    print(f"Test examples:  {len(test_examples)}")

    # Build vocab from training data only
    stoi, itos = build_vocab(train_examples, min_freq=3, max_size=30000)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    # Load GloVe
    embedding_matrix = load_glove_embeddings(stoi, args.glove_path, emb_dim=100).to(DEVICE)

    # Build datasets & loaders
    train_dataset = GAPCNNDataset(train_examples, stoi, max_len=args.max_len)
    val_dataset = GAPCNNDataset(val_examples, stoi, max_len=args.max_len)
    test_dataset = GAPCNNDataset(test_examples, stoi, max_len=args.max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Init model
    model = TextCNN(
        embedding_matrix,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        dropout=0.5,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\n====================================================")
    print("Training CNN baseline...\n")

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_acc, _ = eval_model(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

    # Load best model on val
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(DEVICE)

    # Final evaluation
    print("\n====================================================")
    print("Evaluating on validation set...")
    val_loss, val_acc, val_gender = eval_model(model, val_loader, criterion)

    print("Validation Results:")
    print(f"  Loss:      {val_loss:.4f}")
    print(f"  Accuracy:  {val_acc:.4f}")
    if val_gender is not None:
        print(
            f"  Male pronoun accuracy:   {val_gender['male_acc']:.4f} "
            f"({val_gender['male_total']} examples)"
        )
        print(
            f"  Female/other accuracy:   {val_gender['female_acc']:.4f} "
            f"({val_gender['female_total']} examples)"
        )
        print(f"  Gender bias (M - F):     {val_gender['bias']:.4f}")

    print("\nEvaluating on test set...")
    test_loss, test_acc, test_gender = eval_model(model, test_loader, criterion)

    print("Test Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    if test_gender is not None:
        print(
            f"  Male pronoun accuracy:   {test_gender['male_acc']:.4f} "
            f"({test_gender['male_total']} examples)"
        )
        print(
            f"  Female/other accuracy:   {test_gender['female_acc']:.4f} "
            f"({test_gender['female_total']} examples)"
        )
        print(f"  Gender bias (M - F):     {test_gender['bias']:.4f}")
        # ---- Save embeddings + vocab for WEAT in Colab ----
    save_path = "cnn_glove_embeddings.pt"
    torch.save(
        {
            "stoi": stoi,
            "embeddings": model.embedding.weight.detach().cpu()
        },
        save_path
    )
    print(f"\nSaved CNN embeddings for WEAT to: {save_path}")

    print("\n==================== SUMMARY ====================")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    if test_gender is not None:
        print(f"Test Gender Bias:    {test_gender['bias']:.4f}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
