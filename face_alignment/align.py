import sys
import os

from face_alignment import mtcnn
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime

_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn_model = mtcnn.MTCNN(device=_device, crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

# def get_aligned_face(image_path, rgb_pil_image=None):
#     if rgb_pil_image is None:
#         img = Image.open(image_path).convert('RGB')
#     else:
#         assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
#         img = rgb_pil_image
#     # find face
#     try:
#         bboxes, faces = mtcnn_model.align_multi(img, limit=1)
#         face = faces[0]
#     except Exception as e:
#         print('Face detection Failed due to error.')
#         print(e)
#         face = None

#     return face



def get_aligned_face(image_path, rgb_pil_image=None, multiple=False, limit=None):
    """
    Detect and align one or more faces.
    Args:
        image_path: str - path to image
        rgb_pil_image: optional PIL image (if already loaded)
        multiple: bool - if True, return all aligned faces
        limit: int or None - maximum number of faces to return
    Returns:
        - A single aligned face (PIL image) if multiple=False
        - A list of aligned faces (PIL images) if multiple=True
    """
    if rgb_pil_image is None:
        img = Image.open(image_path).convert("RGB")
    else:
        assert isinstance(rgb_pil_image, Image.Image), \
            "Expected a PIL.Image.Image or image path"
        img = rgb_pil_image

    try:
        # Let limit=None for unlimited or set manually
        bboxes, faces = mtcnn_model.align_multi(img, limit=limit)
        if not faces:
            return None
    except Exception as e:
        print("❌ Face alignment failed:", e)
        return None

    if multiple:
        return faces  # list of aligned faces
    else:
        return faces[0]



