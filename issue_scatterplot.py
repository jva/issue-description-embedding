"""
Embed issue descriptions and project them to 2-D with t-SNE.

USAGE
-----
    python embed_tsne.py issues.csv [model-name]

  • `issues.csv` must have two columns: issue-key, description
  • `model-name` (optional) defaults to 'BAAI/bge-large-en-v1.5'.
    Other good choices:
        - 'nomic-ai/nomic-embed-text-v1.5'
        - 'nvidia/NV-Embed-v2'  (needs instruction prefix, see NOTE)

REQUIREMENTS
------------
pip install -U sentence-transformers pandas scikit-learn torch==2.2.*
"""
import sys, pathlib
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import torch


def load_model(name: str) -> SentenceTransformer:
    """Load the embedding model on Apple-GPU if available, else CPU."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"▶ Loading {name} on {device} …")
    return SentenceTransformer(name, device=device)


def embed_texts(model: SentenceTransformer, texts, batch_size: int = 32):
    """Return L2-normalised embeddings for a list of strings."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def main(csv_path: str, model_name: str):
    in_path  = pathlib.Path(csv_path)
    out_path = in_path.with_stem(in_path.stem + "-output")

    # ------------ 1. Read input ------------------------------------------------
    df = pd.read_csv(in_path, header=None, skiprows=1)
    issue_keys     = df.iloc[:, 0].tolist()
    issue_texts    = df.iloc[:, 1].fillna("").tolist()  # guard against NaNs
    print("input reading - done")

    # ------------ 2. Embeddings ------------------------------------------------
    model     = load_model(model_name)
    vectors   = embed_texts(model, issue_texts)
    vectors   = np.asarray(vectors, dtype=np.float32)
    print("vectors - done")

    # ------------ 3. 2-D projection -------------------------------------------
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", verbose=1)
    coords = tsne.fit_transform(vectors)

    # ------------ 4. Save ------------------------------------------------------
    out_df = pd.DataFrame(
        {"issue-key": issue_keys, "x-coordinate": coords[:, 0], "y-coordinate": coords[:, 1]}
    )
    out_df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"✅ Written → {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python embed_tsne.py issues.csv [model-name]")
    csv_file   = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "BAAI/bge-large-en-v1.5"
    main(csv_file, model_name)
