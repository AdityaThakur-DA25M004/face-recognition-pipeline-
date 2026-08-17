import os
import cv2
import torch
import uuid
import numpy as np
from face_detector import FaceDetector
from face_alignment import align  # your align module
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import argparse
import net  # AdaFace backbone loader

from faiss_store import l2_normalize, load_or_create_langchain_faiss

# -------------------------------
# 🔧 AdaFace setup
# -------------------------------
adaface_ckpt = "./pretrained/adaface_ir50_webface4m.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_adaface_model(architecture="ir_50", ckpt_path=adaface_ckpt):
    model = net.build_model(architecture)
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(model_state)
    model.eval().to(device)
    return model

def to_input(img_rgb):
    """Convert aligned RGB image to AdaFace tensor format."""
    np_img = np.array(img_rgb)
    bgr_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5  # RGB→BGR, normalize to [-1,1]
    tensor = torch.tensor([bgr_img.transpose(2, 0, 1)]).float()
    return tensor

# -------------------------------
# 🔹 Image processing
# -------------------------------
def process_image(img_path, vectorstore, adaface_model, detector, threshold=0.75, save_new_faces_dir="./face_database"):
    """Detect → align → embed → check duplicates in FAISS."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        return vectorstore

    faces = detector.detect_faces(img_path)
    if not faces:
        print(f"⚠️ No faces detected in {img_path}")
        return vectorstore

    aligned_rgb_imgs = align.get_aligned_face(img_path, multiple=True)
    if not aligned_rgb_imgs:
        print(f"⚠️ No valid faces aligned in {img_path}")
        return vectorstore

    embeddings = []
    for aligned_rgb in aligned_rgb_imgs:
        img_tensor = to_input(aligned_rgb).to(device)
        with torch.no_grad():
            feature, _ = adaface_model(img_tensor)
        emb = feature.cpu().numpy().flatten()
        embeddings.append(emb)

    embeddings = np.stack(embeddings)
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
                print(f"⚠️ Duplicate — matches FaceID: {existing_doc.metadata['face_id']} (sim={similarity:.2f})")
                continue

        # ✅ Add new face
        new_doc = Document(
            page_content="",
            metadata={"face_id": new_face_id, "source_image": os.path.basename(img_path)},
        )
        vectorstore.index.add(np.expand_dims(emb, axis=0))
        new_id = str(len(vectorstore.docstore))
        vectorstore.docstore[new_id] = new_doc
        vectorstore.index_to_docstore_id[vectorstore.index.ntotal - 1] = new_id

        print(f"✅ Added new face with FaceID: {new_face_id}")

        # Save aligned image
        save_path = os.path.join(save_new_faces_dir, f"{new_face_id}.jpg")
        os.makedirs(save_new_faces_dir, exist_ok=True)
        aligned_bgr = cv2.cvtColor(np.array(aligned_rgb), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, aligned_bgr)
        print(f"💾 Saved new face to: {save_path}")

    return vectorstore

# -------------------------------
# 🚀 Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaFace + FAISS Duplicate Detection")
    parser.add_argument("--input", default="./inference_image", type=str,
                        help="Path to image or folder containing images.")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Cosine similarity threshold for duplicate check.")
    args = parser.parse_args()

    save_dir = "./embeddings/adaface_langchain_faiss_store"
    os.makedirs(save_dir, exist_ok=True)

    # Load models
    detector = FaceDetector(model_path="./weights/Resnet50_Final.pth")
    adaface_model = load_adaface_model("ir_50")

    # Load FAISS vectorstore (bootstraps an empty one on first run)
    vectorstore = load_or_create_langchain_faiss(save_dir, embedding_dim=512)

    # Process input
    if os.path.isdir(args.input):
        images = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    else:
        images = [args.input]

    for img_path in images:
        vectorstore = process_image(
            img_path, vectorstore, adaface_model, detector, threshold=args.threshold
        )

    # ✅ Save updated FAISS store
    vectorstore.save_local(save_dir)
    print(f"💾 LangChain FAISS store updated and saved to {save_dir}")
