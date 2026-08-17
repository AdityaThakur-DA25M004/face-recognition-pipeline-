import torch
import cv2
import os
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image

# import custom adaface loading utilities
import net
from face_alignment import align

# ----------------------------
# 1️⃣ Load AdaFace model
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

adaface_models = {
    'ir_50': "./pretrained/adaface_ir50_webface4m.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    state_dict = torch.load(adaface_models[architecture], map_location=device)['state_dict']
    model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
    model.load_state_dict(model_state)
    model.eval().to(device)
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5  # RGB → BGR, normalize to [-1,1]
    tensor = torch.tensor([bgr_img.transpose(2, 0, 1)]).float()
    return tensor

# Load AdaFace model
model = load_pretrained_model('ir_50')

# ----------------------------
# 2️⃣ Input/output setup
# ----------------------------
input_dir = "./cropped_images"
output_csv = "./embeddings/adaface_embeddings.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
print(os.getcwd())
# ----------------------------
# 3️⃣ Generate and save embeddings
# ----------------------------
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["face_id", "embedding"])

    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        face_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(input_dir, img_name)

        # Align the face using your face_alignment module
        try:
            aligned_rgb_img = align.get_aligned_face(img_path)
        except Exception as e:
            print(f"⚠️ Skipping {img_name}: face alignment failed ({e})")
            continue

        if aligned_rgb_img is None:
            print(f"⚠️ No face detected in {img_name}, skipping.")
            continue

        # Convert aligned RGB image → normalized tensor
        img_tensor = to_input(aligned_rgb_img).to(device)

        # Generate embedding
        with torch.no_grad():
            feature, _ = model(img_tensor)
            embedding = feature.cpu().numpy().flatten()

        # Normalize (L2)
        embedding = embedding / np.linalg.norm(embedding)

        writer.writerow([face_id, embedding.tolist()])

print(f"✅ AdaFace embeddings saved to {output_csv}")
