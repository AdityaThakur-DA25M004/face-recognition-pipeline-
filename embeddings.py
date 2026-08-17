import torch
from facenet_pytorch import InceptionResnetV1
import cv2
import os
import numpy as np
import csv
from tqdm import tqdm

# ----------------------------
# 1️⃣ Load VGG-Face model (InceptionResnetV1 pretrained on VGGFace2)
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ----------------------------
# 2️⃣ Input/output setup
# ----------------------------
input_dir = "./cropped_images"
output_csv = "./embeddings/vggface_embeddings.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ----------------------------
# 3️⃣ Generate and save embeddings
# ----------------------------
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["face_id", "embedding"])

    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        face_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable file: {img_name}")
            continue

        # Convert to RGB & resize (expected size 160x160)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))

        # Convert to tensor (1, 3, 160, 160)
        img = np.transpose(img, (2, 0, 1)) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()

        # Normalize (L2 norm)
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.flatten().tolist()

        writer.writerow([face_id, embedding])

print(f"✅ VGG-Face embeddings saved to {output_csv}")
