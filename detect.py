from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import csv
from tqdm import tqdm
import uuid
parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=not torch.cuda.is_available(), help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.9, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.9, type=float, help='visualization_threshold')
parser.add_argument("--input_folder",default="./test_images",type=str,help="Folder Containing test images")
parser.add_argument("--output_file",default="./detection/detections.csv",type=str,help="Output Folder containing csv of Images detections file")
parser.add_argument("--output_crop",default="./cropped_images",type=str,help="Store cropped images for face embedding")
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    os.makedirs(os.path.dirname(args.output_file),exist_ok=True)
    csv_file = open(args.output_file,"w",newline='')
    csv_writer = csv.writer(csv_file)
    header = ["image_id","image_path","face_id","conf","x1","y1","x2","y2",
              "lmle_x","lmle_y","lmre_x","lmre_y","lmn_x","lmn_y",
              "lmlm_x","lmlm_y","lmrm_x","lmrm_y","yaw_deg"]
    
    csv_writer.writerow(header)

    image_list=[f for f in os.listdir(args.input_folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    print(f"Processing {len(image_list)} images...")


    os.makedirs(args.output_crop,exist_ok=True)
    def align_face(img, landms):
            """Aligns and normalizes a face crop using 5 landmarks (for VGG-Face)."""
            # Reference 5 points from VGG-Face template (224x224 crop)
            ref_pts = np.float32([
                [96.0, 112.0],   # left eye
                [160.0, 112.0],  # right eye
                [128.0, 160.0],  # nose
                [104.0, 200.0],  # left mouth
                [152.0, 200.0]   # right mouth
            ])
            dst_size = (224, 224)

            # Compute similarity transform
            src_pts = landms.reshape((5, 2)).astype(np.float32)
            M = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)[0]
            aligned = cv2.warpAffine(img, M, dst_size, borderValue=0.0)

            # 🔹 VGG-Face normalization (subtract BGR mean)
            aligned = aligned.astype(np.float32)
            mean_bgr = np.array([93.5940, 104.7624, 129.1863], dtype=np.float32)
            aligned -= mean_bgr

            # return as (3, 224, 224) tensor-like NumPy array
            aligned = np.transpose(aligned, (2, 0, 1))
            return aligned

    

    def estimate_yaw(landms):
        """
        Roughly estimate yaw (left/right rotation) from 5 facial landmarks.
        Positive yaw = turned right, Negative = turned left.
        Returns yaw angle in degrees.
        """
        le, re, n, lm, rm = landms.reshape((5, 2))

        # horizontal distance between eyes
        eye_dx = re[0] - le[0]
        if abs(eye_dx) < 1e-3:
            return 0.0

        # how far nose tip is shifted from eye center
        nose_center_x = n[0] - (le[0] + re[0]) / 2
        yaw_ratio = nose_center_x / eye_dx

        # small-angle approximation to get degrees
        yaw_deg = np.degrees(np.arctan(yaw_ratio))
        return yaw_deg



    total_faces = 0
    for idx, img_name in enumerate(tqdm(image_list)):
        img_path = os.path.join(args.input_folder, img_name)
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            print(f"Skipping {img_name} (cannot read)")
            continue

        # === preprocessing ===
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        # === inference ===
        loc, conf, landms = net(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([im_width, im_height] * 5).to(device)
        landms = (landms * scale1 / resize).cpu().numpy()

        inds = np.where(scores > args.confidence_threshold)[0]
        if inds.size == 0:
            print(f"No faces detected in {img_name}")
            continue
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

        order = scores.argsort()[::-1][:args.top_k]
        boxes, landms, scores = boxes[order], landms[order], scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets, landms = dets[keep, :], landms[keep]
        dets, landms = dets[:args.keep_top_k, :], landms[:args.keep_top_k, :]

        face_count = 0
        for i in range(dets.shape[0]):
            conf_i = float(dets[i, 4])
            if conf_i < args.vis_thres:
                continue

            x1, y1, x2, y2 = map(int, dets[i, :4])
            w, h = x2 - x1, y2 - y1
            if w < 40 or h < 40:
                continue

            face_landm = landms[i]
            yaw = estimate_yaw(face_landm)
            if abs(yaw) > 35:  # you can adjust this threshold (30–40 is typical)
                print(f"Skipping sideways face (yaw={yaw:.1f}°)")
                continue

            face_id = str(uuid.uuid4())[:8]
            aligned_face = align_face(img_raw, face_landm)

            crop_path = os.path.join(args.output_crop, f"{face_id}.jpg")
            # Convert back for saving
            save_face = np.transpose(aligned_face, (1, 2, 0)) + np.array([93.5940, 104.7624, 129.1863])
            save_face = np.clip(save_face, 0, 255).astype(np.uint8)
            cv2.imwrite(crop_path, save_face)
            csv_writer.writerow([
                idx,
                img_path,
                face_id,
                conf_i,
                float(dets[i, 0]), float(dets[i, 1]), float(dets[i, 2]), float(dets[i, 3]),
                *face_landm[:10].astype(float),
                yaw  # optional: store yaw in CSV
            ])

            face_count += 1
            total_faces += 1

        print(f"→ {img_name}: {face_count} faces saved")

    print(f"\nCompleted! Total faces saved: {total_faces}")
    csv_file.close()
    print("Detections CSV:", args.output_file, "Crops folder:", args.output_crop)
