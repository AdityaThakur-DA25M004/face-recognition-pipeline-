import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def save_faiss_with_metadata(index_path,ids_path,save_dir="./embeddings/langchain_faiss_store"):

    print("💎 Loading FAISS index and face IDs...")

    index= faiss.read_index(index_path)
    face_ids=np.load(ids_path,allow_pickle=True)

    if index.ntotal!=len(face_ids):
        raise ValueError(f"❌ Mismatch : {index.ntotal} vectors in FAISS but {len(face_ids)} IDS found")
    print(f"✅ Loaded {index.ntotal} embeddings and {len(face_ids)} face IDs")


    docs = [
        Document(page_content="",metadata={"face_id":fid}) for fid in face_ids
        ]
    vectorstore=FAISS(
        embedding_function=None,
        index=index,
        docstore={},
        index_to_docstore_id={}
    )
    vectorstore.docstore={str(i):docs[i] for i in range(len(docs))}
    vectorstore.index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    vectorstore.save_local(save_dir)
    print(f"💾 LangChain FAISS store with metadata saved to: {save_dir}")


def main():
    # ----------------------------
    # Input paths (edit these if needed)
    # ----------------------------
    index_path = "./embeddings/faiss.index"
    ids_path = "./embeddings/face_ids.npy"
    save_dir = "./embeddings/langchain_faiss_store"

    save_faiss_with_metadata(index_path, ids_path, save_dir)


if __name__ == "__main__":
    main()
