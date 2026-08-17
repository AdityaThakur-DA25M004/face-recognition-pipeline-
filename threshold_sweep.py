"""
threshold_sweep.py
-------------------
Compute embeddings once for a gallery/probe evaluation set (see
build_lfw_eval_set.py / evaluation.py for the expected person_XXX/ layout),
then sweep similarity thresholds to report the precision/recall/F1 tradeoff
and optionally plot a precision-recall-vs-threshold curve.

Usage:
    python threshold_sweep.py --data ./lfw_evaluation_images --plot ./logs/pr_curve.png
"""

import argparse
import os

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from face_aligner import FaceAlignerVGG
from face_detector import FaceDetector


def l2_norm_vec(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    n = np.linalg.norm(x)
    return x / n if n != 0 else x


def get_embedding(path, detector, aligner, embedder, device):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    faces = detector.detect_faces(path)
    if not faces:
        return None
    aligned = aligner.align_and_normalize(img_bgr, faces[0]["landmarks"])
    if aligned is None:
        return None
    aligned = aligned.to(device)
    with torch.no_grad():
        feat = embedder(aligned).cpu().numpy().flatten()
    return l2_norm_vec(feat)


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    detector = FaceDetector(model_path=args.weights)
    aligner = FaceAlignerVGG(device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    persons = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))

    gallery_emb, gallery_ids = [], []
    probes = []  # (person_id, embedding or None)

    for pid in persons:
        folder = os.path.join(args.data, pid)
        imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not imgs:
            continue
        emb = get_embedding(os.path.join(folder, imgs[0]), detector, aligner, embedder, device)
        if emb is None:
            continue
        gallery_emb.append(emb)
        gallery_ids.append(pid)
        for probe_name in imgs[1:]:
            probes.append((pid, get_embedding(os.path.join(folder, probe_name), detector, aligner, embedder, device)))

    gallery_emb = np.stack(gallery_emb).astype("float32")
    print(f"Gallery: {len(gallery_ids)} identities, Probes: {len(probes)}")

    thresholds = args.thresholds
    rows = []
    for threshold in thresholds:
        TP = FP = FN = 0
        for pid, emb in probes:
            if emb is None:
                FN += 1
                continue
            sims = gallery_emb @ emb
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            matched = gallery_ids[idx]
            if sim >= threshold:
                TP += 1 if matched == pid else 0
                FP += 1 if matched != pid else 0
            else:
                FN += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = FP / len(probes) if probes else 0.0
        rows.append((threshold, precision, recall, f1, fpr))

    header = f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>8} {'FPR':>8}"
    print(header)
    lines = [header]
    for threshold, precision, recall, f1, fpr in rows:
        line = f"{threshold:>10.2f} {precision*100:>9.1f}% {recall*100:>9.1f}% {f1*100:>7.1f}% {fpr*100:>7.2f}%"
        print(line)
        lines.append(line)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"Gallery: {len(gallery_ids)} identities, Probes: {len(probes)}\n")
            f.write("\n".join(lines) + "\n")
        print(f"Report saved to: {args.output}")

    if args.plot:
        import matplotlib.pyplot as plt

        ts = [r[0] for r in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(ts, [r[1] * 100 for r in rows], marker="o", label="Precision")
        plt.plot(ts, [r[2] * 100 for r in rows], marker="o", label="Recall")
        plt.plot(ts, [r[3] * 100 for r in rows], marker="o", label="F1")
        plt.xlabel("Cosine similarity threshold")
        plt.ylabel("%")
        plt.title("Precision / Recall / F1 vs. threshold (LFW eval)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plt.savefig(args.plot, dpi=150)
        print(f"Plot saved to: {args.plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep similarity thresholds on a gallery/probe eval set.")
    parser.add_argument("--data", type=str, default="./lfw_evaluation_images")
    parser.add_argument("--weights", type=str, default="./weights/Resnet50_Final.pth")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    parser.add_argument("--output", type=str, default="./logs/threshold_sweep_lfw.txt")
    parser.add_argument("--plot", type=str, default="./logs/pr_curve_lfw.png")
    parser.add_argument("--force-cpu", dest="force_cpu", action="store_true")
    main(parser.parse_args())
