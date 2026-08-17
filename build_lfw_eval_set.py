"""
build_lfw_eval_set.py
----------------------
Build a gallery/probe evaluation set from the real LFW (Labeled Faces in the
Wild) dataset, in the person_XXX/ folder layout expected by evaluation.py.

Unlike ./evaluation_images (synthetic crop/rotate/brightness augmentations of
a single photo per person), this uses genuinely different photographs of the
same real person - different pose, lighting, and expression - which is a
meaningfully harder and more realistic face-verification benchmark.

LFW images are pulled from a public Hugging Face mirror of the dataset
(logasja/lfw). LFW is distributed for non-commercial research/benchmarking
use; this script only fetches images transiently to build a local eval set
and does not redistribute them - the output folder is gitignored.

Usage:
    python build_lfw_eval_set.py --num-identities 60 --min-images 4 --max-images 6
"""

import argparse
import io
import os
import random

import pandas as pd
from PIL import Image

PARQUET_URL = "https://huggingface.co/api/datasets/logasja/lfw/parquet/default/train/0.parquet"


def main(args):
    cache_path = args.cache
    if not os.path.exists(cache_path):
        print(f"Downloading LFW parquet to {cache_path} ...")
        import urllib.request
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(PARQUET_URL, cache_path)

    df = pd.read_parquet(cache_path)
    counts = df["label"].value_counts()
    candidates = counts[(counts >= args.min_images) & (counts <= args.max_images * 2)].index.tolist()

    random.seed(args.seed)
    random.shuffle(candidates)
    chosen = candidates[: args.num_identities]

    os.makedirs(args.output, exist_ok=True)
    total = 0
    for label in chosen:
        rows = df[df["label"] == label].reset_index(drop=True)
        imgs = rows["image"].tolist()[: args.max_images]
        person_dir = os.path.join(args.output, f"person_{label}")
        os.makedirs(person_dir, exist_ok=True)
        for i, img_dict in enumerate(imgs):
            img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
            fname = "img1.jpg" if i == 0 else f"img_dup{i}.jpg"
            img.save(os.path.join(person_dir, fname), quality=95)
            total += 1

    print(f"Wrote {total} real images across {len(chosen)} identities to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a real-photo gallery/probe eval set from LFW.")
    parser.add_argument("--num-identities", type=int, default=60)
    parser.add_argument("--min-images", type=int, default=4, help="Minimum images per identity to include")
    parser.add_argument("--max-images", type=int, default=6, help="Max images kept per identity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./lfw_evaluation_images")
    parser.add_argument("--cache", type=str, default="./data/lfw_train.parquet", help="Local cache for the downloaded parquet")
    main(parser.parse_args())
