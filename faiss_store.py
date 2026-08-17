"""
Shared LangChain/FAISS vector-store helpers used by face_pipeline.py and
face_pipeline_adaface.py.
"""

import os
import pickle

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_langchain_faiss(save_dir: str) -> FAISS:
    """Load a LangChain FAISS store that was saved without an embedding function."""
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory not found: {save_dir}")

    print(f"Loading LangChain FAISS store from: {save_dir}")
    vectorstore = FAISS.load_local(
        save_dir,
        embeddings=None,
        allow_dangerous_deserialization=True,  # required because no embedding function
    )
    print(f"Loaded LangChain FAISS with {len(vectorstore.docstore)} faces.")
    return vectorstore


def load_or_create_langchain_faiss(save_dir: str, embedding_dim: int) -> FAISS:
    """Load an existing store, or bootstrap an empty one on first run."""
    if os.path.exists(os.path.join(save_dir, "index.faiss")):
        return load_langchain_faiss(save_dir)

    print(f"No existing store at {save_dir} — creating a new empty index (dim={embedding_dim}).")
    index = faiss.IndexFlatIP(embedding_dim)
    vectorstore = FAISS(
        embedding_function=None,
        index=index,
        docstore={},
        index_to_docstore_id={},
    )
    return vectorstore


def save_langchain_faiss(vectorstore: FAISS, index_path: str, meta_path: str) -> None:
    faiss.write_index(vectorstore.index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "docstore": vectorstore.docstore,
                "index_to_docstore_id": vectorstore.index_to_docstore_id,
            },
            f,
        )
    print(f"LangChain FAISS updated with {vectorstore.index.ntotal} faces.")
