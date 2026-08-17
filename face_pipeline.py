import os
import cv2
import torch
import uuid
import numpy as np
from facenet_pytorch import InceptionResnetV1
from face_detector import FaceDetector
from face_aligner import FaceAlignerVGG
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import argparse

from faiss_store import l2_normalize, load_langchain_faiss


def process_image(img_path, vectorstore, embedder, detector, aligner, threshold=0.75,save_new_faces_dir="./face_database"):
    """Detect faces → align → embed → check duplicates in LangChain FAISS."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        return vectorstore

    faces = detector.detect_faces(img_path)
    if not faces:
        print(f"⚠️ No faces detected in {img_path}")
        return vectorstore

    batch = aligner.align_multiple_faces(img, faces, save_dir="./aligned_faces")
    if batch is None:
        print(f"⚠️ No valid faces aligned in {img_path}")
        return vectorstore

    with torch.no_grad():
        embeddings = embedder(batch).cpu().numpy()
    embeddings = l2_normalize(embeddings)

    for i, emb in enumerate(embeddings):
        new_face_id = str(uuid.uuid4())[:8]
        emb = emb.astype(np.float32)

        # 🔹 Search for duplicates
        if vectorstore.index.ntotal > 0:
            D, I = vectorstore.index.search(np.expand_dims(emb, axis=0), k=1)
            similarity = D[0][0]
            if similarity > threshold:
                existing_doc_id = vectorstore.index_to_docstore_id[I[0][0]]
                existing_doc = vectorstore.docstore[existing_doc_id]
                print(f"⚠️ Duplicate image — already exists with FaceID: {existing_doc.metadata['face_id']} (sim={similarity:.2f})")
                continue

        # ✅ Add unique face
        new_doc = Document(
            page_content="",
            metadata={"face_id": new_face_id, "source_image": os.path.basename(img_path)},
        )
        vectorstore.index.add(np.expand_dims(emb, axis=0))
        new_id = str(len(vectorstore.docstore))
        vectorstore.docstore[new_id] = new_doc
        vectorstore.index_to_docstore_id[vectorstore.index.ntotal - 1] = new_id

        print(f"✅ Added new face with FaceID: {new_face_id}")
        # 🔹 Save aligned face image to database folder
        aligned_face_tensor = batch[i].cpu().numpy()  # shape: (3,160,160)
        aligned_face = np.transpose(aligned_face_tensor, (1, 2, 0))  # (H,W,C)
        aligned_face = (aligned_face * 255).astype(np.uint8)
        save_path = os.path.join(save_new_faces_dir, f"{new_face_id}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
        print(f"💾 Saved new face image to: {save_path}")

    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face pipeline with LangChain FAISS")
    parser.add_argument("--input",default="./inference_image", type=str,
                        help="Path to image or folder containing images.")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Similarity threshold for duplicate check (default=0.75).")
    args = parser.parse_args()

    # Paths
    save_dir = "./embeddings/langchain_faiss_store"

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FaceDetector(model_path="./weights/Resnet50_Final.pth")
    aligner = FaceAlignerVGG(device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Load FAISS vectorstore
    vectorstore = load_langchain_faiss(save_dir)

    # Process input
    if os.path.isdir(args.input):
        images = [os.path.join(args.input, f) for f in os.listdir(args.input)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        images = [args.input]

    for img_path in images:
        vectorstore = process_image(
            img_path, vectorstore, embedder, detector, aligner, threshold=args.threshold
        )

    # Save updated FAISS store
    vectorstore.save_local(save_dir)
    print(f"LangChain FAISS store updated and saved to {save_dir}")



