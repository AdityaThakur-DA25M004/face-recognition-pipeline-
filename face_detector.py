# Newer Version
from facenet_pytorch import MTCNN
import os
import torch
import numpy as np
import cv2
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

import matplotlib.pyplot as plt
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    print(f"Loaded {len(used_pretrained_keys)} matching keys")
    assert len(used_pretrained_keys) > 0, "No matching Keys found"
    return True


def remove_prefix(state_dict, prefix):
    print(f"Removing prefix {prefix}")
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print(f"🔹 Loading pretrained model from {pretrained_path}")
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict:
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def estimate_yaw(landms):
    le, re, n, lm, rm = landms.reshape((5, 2))
    eye_dx = re[0] - le[0]
    if abs(eye_dx) < 1e-3:
        return 0.0
    nose_center_x = n[0] - (le[0] + re[0]) / 2
    yaw_ratio = nose_center_x / eye_dx
    return np.degrees(np.arctan(yaw_ratio))


# 🔹 ADD THIS POST-FILTER FUNCTION
def remove_overlapping_faces(faces, iou_threshold=0.5):
    """
    Removes overlapping or duplicate face detections based on IoU threshold.
    Keeps the one with higher confidence.
    """
    if len(faces) <= 1:
        return faces

    faces = sorted(faces, key=lambda x: x["confidence"], reverse=True)
    final_faces = []

    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    for f in faces:
        keep = True
        for kept in final_faces:
            if iou(f["bbox"], kept["bbox"]) > iou_threshold:
                keep = False
                break
        if keep:
            final_faces.append(f)

    if len(faces) != len(final_faces):
        print(f"🔹 Post-filter removed {len(faces) - len(final_faces)} overlapping detections")

    return final_faces


class FaceDetector:
    def __init__(
        self,
        model_path: str,
        network: str = "resnet50",
        confidence_threshold: float = 0.95,
        nms_threshold: float = 0.25,  # 🔹 Slightly lower for aggressive NMS
        vis_threshold: float = 0.95,
        use_cpu: bool = None
    ):
        if use_cpu is None:
            use_cpu = not torch.cuda.is_available()
        self.cfg = cfg_mnet if network == "mobile0.25" else cfg_re50
        self.device = torch.device("cpu" if use_cpu else "cuda")
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_threshold
        self.vis_thresh = vis_threshold  # 🔹 Fixed typo (was vis_thrsh)

        self.net = RetinaFace(cfg=self.cfg, phase="test")
        self.net = load_model(self.net, model_path, use_cpu)
        self.net.eval()
        self.net.to(self.device)

        print("✅ RetinaFace model initialized and ready.")
    

    def detect_faces(self, image_path: str):
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            raise FileNotFoundError(f"❌ Cannot read image: {image_path}")

        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        loc, conf, landms = self.net(img)

        prior_box = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = prior_box.forward().to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        resize = 1
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), priors.data, self.cfg["variance"])
        scale1 = torch.Tensor([im_width, im_height] * 5).to(self.device)
        landms = (landms * scale1 / resize).cpu().numpy()

        inds = np.where(scores > self.conf_thresh)[0]
        if len(inds) == 0:
            print("⚠️ No faces detected.")
            return []

        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

        # 🔹 Standard NMS
        order = scores.argsort()[::-1]
        boxes, landms, scores = boxes[order], landms[order], scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, self.nms_thresh)
        dets, landms = dets[keep, :], landms[keep]

        # 🔹 Collect faces
        verifier = MTCNN(keep_all=False, device=self.device)

        faces = []
        for i in range(dets.shape[0]):
            conf_i = float(dets[i, 4])
            if conf_i < self.vis_thresh:
                continue

            bbox = list(map(int, dets[i, :4]))
            # x1, y1, x2, y2 = bbox
            # w, h = x2 - x1, y2 - y1
            # if w < 40 or h < 40:
            #     continue

            face_landm = landms[i]
            # lm = face_landm.reshape(5, 2)

            # # geometric consistency filter
            # le, re, n, lm_m, rm_m = lm
            # eye_dist = np.linalg.norm(re - le)
            # mouth_width = np.linalg.norm(rm_m - lm_m)
            # if eye_dist < 20 or mouth_width / eye_dist < 0.5 or mouth_width / eye_dist > 2.0:
            #     continue

            # # flat landmark filter
            # if np.std(lm[:, 1]) < 5:
            #     continue

            # # aspect ratio filter
            # aspect_ratio = w / float(h)
            # if aspect_ratio > 1.2 or aspect_ratio < 0.5:
            #     continue

            yaw = estimate_yaw(face_landm)
            # if abs(yaw) > 35:
            #     continue

            face_id = str(i + 1)
            faces.append({
                "face_id": face_id,
                "bbox": bbox,
                "confidence": conf_i,
                "landmarks": face_landm,
                "yaw_deg": yaw,
            })

        # 🔹 Apply post-filter for overlapping faces
        faces = remove_overlapping_faces(faces, iou_threshold=0.45)

        print(f"✅ {len(faces)} unique faces detected in {os.path.basename(image_path)}")
        return faces
