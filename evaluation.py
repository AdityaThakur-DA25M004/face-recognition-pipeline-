#!/usr/bin/env python3
"""
evaluate_duplicates_gallery.py

Evaluate duplicate-detection performance using a folder-per-person dataset.

Assumptions (dataset layout):
./evaluation_images/
    person_001/
        img1.jpg        <-- enrolled image (gallery)
        img_dup1.jpg    <-- probe (duplicate)
        img_dup2.jpg
    person_002/
        img1.jpg
        img_dup1.jpg
    ...

Method:
 1. Enroll the FIRST image from each person folder into an in-memory FAISS gallery.
 2. For every other image (probes), extract embedding and search gallery (top-1).
 3. Decide:
    - predicted_duplicate = (similarity >= threshold)
    - If predicted_duplicate and matched_gallery_id == probe_person -> TP
    - If predicted_duplicate and matched_gallery_id != probe_person -> FP
    - If not predicted_duplicate -> FN (a missed duplicate)
    - If detection/alignment fails -> count as FN (missed)
 4. Compute Precision, Recall, F1, Detection success rate, Avg inference time, FPR (w.r.t probes).

Usage:
    python evaluate_duplicates_gallery.py \
        --data ./evaluation_images \
        --weights ./weights/Resnet50_Final.pth \
        --threshold 0.75 \
        --output ./logs/duplicate_eval_report.txt

This script **does not** modify your LangChain/FAISS store used by the pipeline.
It uses the same detector/aligner/embedder modules so results are consistent.
"""

import os
import time
import argparse
import numpy as np
import faiss
import torch
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import cv2

# Import your local wrappers (adjust names if different)
from face_detector import FaceDetector
from face_aligner import FaceAlignerVGG

# -------------------------
# Helpers
# -------------------------
def l2_norm_vec(x: np.ndarray):
    x = x.astype('float32')
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n

def get_embedding_from_image(path, detector, aligner, embedder, device):
    """
    Detect -> align -> embed a single face image.
    Returns: (embedding (1D float32 numpy), status_str)
    status_str: "ok", "read_failed", "no_faces", "align_failed"
    """
    # read image (BGR)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None, "read_failed"

    # detector.detect_faces expects image_path in your detector implementation
    faces = detector.detect_faces(path)
    if not faces:
        return None, "no_faces"

    # use the first detected face for probe/enroll
    face = faces[0]
    aligned_tensor = aligner.align_and_normalize(img_bgr, face["landmarks"])
    if aligned_tensor is None:
        return None, "align_failed"

    # aligned_tensor is (1,3,160,160) on device from aligner implementation
    aligned_tensor = aligned_tensor.to(device)
    with torch.no_grad():
        feat = embedder(aligned_tensor).cpu().numpy().flatten()
    feat = l2_norm_vec(feat)
    return feat.astype('float32'), "ok"

# -------------------------
# Main evaluation
# -------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    # initialize models
    detector = FaceDetector(model_path=args.weights)
    aligner = FaceAlignerVGG(device=device)
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Discover person folders
    persons = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    if len(persons) == 0:
        print("No person folders found in", args.data)
        return

    # Build gallery: enroll first image per person
    gallery_embeddings = []
    gallery_ids = []
    print("🗂️  Building gallery (enrolling first image per person)...")
    for pid in tqdm(persons):
        folder = os.path.join(args.data, pid)
        imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        if not imgs:
            print(f"⚠️  No images in {folder}, skipping.")
            continue
        enroll_path = os.path.join(folder, imgs[0])  # enroll first image
        emb, status = get_embedding_from_image(enroll_path, detector, aligner, embedder, device)
        if emb is None:
            print(f"⚠️  Could not enroll {pid} ({enroll_path}) -> {status}. Skipping identity.")
            continue
        gallery_embeddings.append(emb)
        gallery_ids.append(pid)

    if len(gallery_embeddings) == 0:
        print("No gallery embeddings created. Exiting.")
        return

    D = np.stack(gallery_embeddings).astype('float32')  # (G, dim)
    dim = D.shape[1]
    # FAISS IndexFlatIP on normalized vectors -> inner product equals cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(D)

    # Evaluation counters
    TP = FP = FN = 0
    detection_success = 0
    total_probes = 0
    inference_times = []

    print("\n🔎 Running probes (every image except enrolled ones)...")
    for pid in tqdm(persons):
        folder = os.path.join(args.data, pid)
        imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        if len(imgs) <= 1:
            # no probes for this person
            continue

        # probes are imgs[1:]
        for probe_name in imgs[1:]:
            probe_path = os.path.join(folder, probe_name)
            total_probes += 1
            t0 = time.time()
            emb, status = get_embedding_from_image(probe_path, detector, aligner, embedder, device)
            t_elapsed = time.time() - t0
            inference_times.append(t_elapsed)

            if emb is None:
                # detection/alignment failed -> count as FN (missed duplicate)
                FN += 1
                continue

            detection_success += 1

            emb_q = emb.reshape(1, -1).astype('float32')
            scores, I = index.search(emb_q, 1)  # top-1
            sim = float(scores[0,0])
            idx = int(I[0,0]) if I.size else -1
            matched_person = gallery_ids[idx] if (0 <= idx < len(gallery_ids)) else None

            predicted_duplicate = (sim >= args.threshold)

            if predicted_duplicate:
                if matched_person == pid:
                    TP += 1
                else:
                    FP += 1
            else:
                FN += 1

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    detection_success_rate = (detection_success / total_probes) if total_probes > 0 else 0.0
    avg_inference_ms = (np.mean(inference_times) * 1000) if inference_times else 0.0
    false_positive_rate = (FP / total_probes) if total_probes > 0 else 0.0
    overall_accuracy = TP / total_probes if total_probes > 0 else 0.0  # fraction of probes correctly identified as duplicate

    # Print summary
    print("\n=== Duplicate Detection Evaluation Summary ===")
    print(f"Gallery enrolled identities: {len(gallery_ids)}")
    print(f"Total probes (non-enrolled images): {total_probes}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-score:  {f1*100:.2f}%")
    print(f"Detection success rate (faces detected & aligned): {detection_success_rate*100:.2f}%")
    print(f"Avg inference time per probe: {avg_inference_ms:.2f} ms")
    print(f"False Positive Rate (w.r.t probes): {false_positive_rate*100:.2f}%")
    print(f"Overall probe accuracy (TP/total_probes): {overall_accuracy*100:.2f}%")
    print("==============================================\n")

    # Save report
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write("Duplicate Detection Evaluation Report\n")
        fo.write("=====================================\n")
        fo.write(f"Gallery size: {len(gallery_ids)}\n")
        fo.write(f"Total probes: {total_probes}\n")
        fo.write(f"TP: {TP}\nFP: {FP}\nFN: {FN}\n\n")
        fo.write(f"Precision: {precision*100:.2f}%\n")
        fo.write(f"Recall:    {recall*100:.2f}%\n")
        fo.write(f"F1-score:  {f1*100:.2f}%\n\n")
        fo.write(f"Detection success rate: {detection_success_rate*100:.2f}%\n")
        fo.write(f"Avg inference time (ms): {avg_inference_ms:.2f}\n")
        fo.write(f"False Positive Rate (probes): {false_positive_rate*100:.2f}%\n")
        fo.write(f"Overall probe accuracy: {overall_accuracy*100:.2f}%\n")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate duplicate detection using gallery enrollment (one image per identity).")
    parser.add_argument("--data", type=str, default="./evaluation_images", help="Folder with person_xxx subfolders")
    parser.add_argument("--weights", type=str, default="./weights/Resnet50_Final.pth", help="RetinaFace weights path")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold (cosine/inner-product on normalized vectors)")
    parser.add_argument("--output", type=str, default="./logs/duplicate_eval_report.txt", help="Output report path")
    parser.add_argument("--force-cpu", dest="force_cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()
    main(args)
