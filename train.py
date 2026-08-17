from __future__ import print_function
import os
import math
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import cv2
import numpy as np
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/my_faces/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./runs/finetuned/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder, exist_ok=True)

cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg["epoch"]
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

# --- Load full RetinaFace pretrained weights for fine-tuning ---
if args.network == "mobile0.25":
    pretrained_path = "./weights/mobilenet0.25_Final.pth"
elif args.network == "resnet50":
    pretrained_path = "./weights/Resnet50_Final.pth"
else:
    pretrained_path = None

if pretrained_path and os.path.exists(pretrained_path):
    print(f"Loading pretrained RetinaFace weights from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location="cpu")
    net.load_state_dict(state_dict, strict=False)
else:
    print("No pretrained RetinaFace weights found - training from scratch!")

# --- Resume checkpoint if provided ---
if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

# --- Freeze backbone layers for fine-tuning; unfrozen gradually in train() ---
print("Freezing backbone layers for fine-tuning...")
for name, param in net.body.named_parameters():
    param.requires_grad = False
print(f"Frozen {len([p for p in net.body.parameters() if not p.requires_grad])} backbone layers.")

trainable = [name for name, p in net.named_parameters() if p.requires_grad]
print(f"Trainable layers after freezing: {len(trainable)}")

if device.type == "cuda" and num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).to(device)
else:
    net = net.to(device)

cudnn.benchmark = True

optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, net.parameters()),
    lr=initial_lr * 1.2,
    momentum=momentum,
    weight_decay=weight_decay
)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 3, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def visualize_validation_predictions(
    net,
    cfg,
    val_loader,
    device,
    epoch,
    save_dir="./val_visuals",
    vis_thres=0.6,
    max_images=20,
    rgb_mean=(104, 117, 123),
    conf_threshold=0.98,
):
    """Save a handful of prediction visualizations for a validation epoch."""
    net.eval()
    saved = 0
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                images_batch, targets_batch = batch
            except Exception:
                print("Unexpected batch format in visualize_validation_predictions, skipping batch.")
                continue

            if isinstance(images_batch, (list, tuple)):
                images_batch = torch.stack(images_batch, dim=0)

            images_batch = images_batch.to(device)
            batch_size = images_batch.shape[0]

            locs, confs, landms_pred = net(images_batch)

            for i in range(batch_size):
                if saved >= max_images:
                    return saved

                loc = locs[i:i + 1]
                conf = confs[i:i + 1]
                landm = landms_pred[i:i + 1]
                conf = F.softmax(conf, dim=-1)

                tgt = targets_batch[i]
                tgt_np = tgt.cpu().numpy() if isinstance(tgt, torch.Tensor) else np.array(tgt)

                img_tensor = images_batch[i].detach().cpu().numpy()  # (C,H,W)
                if img_tensor.ndim != 3:
                    print("Unexpected image tensor shape, skipping image")
                    continue
                img_disp = img_tensor.transpose(1, 2, 0)  # H,W,C
                img_disp = img_disp + np.array(rgb_mean, dtype=np.float32)
                img_disp = np.clip(img_disp, 0, 255).astype(np.uint8).copy()

                im_h, im_w = img_disp.shape[:2]
                priorbox = PriorBox(cfg, image_size=(im_h, im_w))
                priors = priorbox.forward().to(device)
                priors_data = priors.data

                try:
                    boxes = decode(loc.data.squeeze(0), priors_data, cfg['variance'])
                    boxes = boxes * torch.Tensor([im_w, im_h, im_w, im_h]).to(device)
                    boxes = boxes.cpu().numpy()
                    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                    landms = decode_landm(landm.data.squeeze(0), priors_data, cfg['variance'])
                    scale1 = torch.Tensor([im_w, im_h] * 5).to(device)
                    landms = (landms * scale1).cpu().numpy()
                except Exception as e:
                    print(f"Decoding failed for idx {i}: {e}")
                    continue

                inds = np.where(scores > vis_thres)[0]
                if inds.size == 0:
                    topk = min(30, boxes.shape[0])
                    order = scores.argsort()[::-1][:topk]
                else:
                    order = inds[scores[inds].argsort()[::-1]]

                boxes_to_draw = boxes[order]
                scores_to_draw = scores[order]
                landms_to_draw = landms[order]

                img_out = img_disp.copy()
                valid_indices = [idx for idx, sc in enumerate(scores_to_draw) if sc >= conf_threshold]

                for idx_p in valid_indices:
                    bb = boxes_to_draw[idx_p].astype(int)
                    sc = scores_to_draw[idx_p]

                    x1, y1, x2, y2 = bb
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(im_w - 1, x2), min(im_h - 1, y2)

                    cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    y_text = max(20, y1 - 5)
                    cv2.putText(img_out, f"{sc:.4f}", (x1, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if landms_to_draw is not None and landms_to_draw.shape[0] > idx_p:
                        lm = landms_to_draw[idx_p].reshape(-1, 2).astype(int)
                        for (lx, ly) in lm:
                            if 0 <= lx < im_w and 0 <= ly < im_h:
                                cv2.circle(img_out, (lx, ly), 2, (255, 0, 0), -1)

                # Draw ground-truth boxes/landmarks in green.
                # Expected tgt_np shape: (num_gt, 15) -> [xmin,ymin,xmax,ymax, landmarks(10), label]
                try:
                    if tgt_np.size > 0:
                        rows = tgt_np if (tgt_np.ndim == 2 and tgt_np.shape[1] >= 5) else tgt_np.reshape(1, -1)
                        for gt in rows:
                            gx1 = int(gt[0] * im_w) if gt[0] <= 1.0 else int(gt[0])
                            gy1 = int(gt[1] * im_h) if gt[1] <= 1.0 else int(gt[1])
                            gx2 = int(gt[2] * im_w) if gt[2] <= 1.0 else int(gt[2])
                            gy2 = int(gt[3] * im_h) if gt[3] <= 1.0 else int(gt[3])
                            cv2.rectangle(img_out, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                            if gt.shape[0] >= 14:
                                lm = gt[4:14].reshape(5, 2)
                                if np.max(lm) <= 1.0:
                                    lm[:, 0] *= im_w
                                    lm[:, 1] *= im_h
                                for (lx, ly) in lm.astype(int):
                                    cv2.circle(img_out, (int(lx), int(ly)), 2, (0, 255, 0), -1)
                except Exception as e:
                    print(f"Failed drawing GT for idx {i}: {e}")

                save_name = os.path.join(epoch_dir, f"img_{batch_idx}_{i}.jpg")
                try:
                    cv2.imwrite(save_name, img_out)
                    saved += 1
                except Exception as e:
                    print(f"Failed to save visualization {save_name}: {e}")

                if saved >= max_images:
                    return saved

    return saved


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Warmup + step-decay LR schedule with a periodic boost, adapted from
    https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    warmup_epochs = 3
    boost_interval = 15
    boost_factor = 1.5
    min_lr = 1e-6
    max_lr = initial_lr * 3.0

    if epoch < warmup_epochs:
        lr = 1e-6 + (initial_lr - 1e-6) * (epoch * epoch_size + iteration) / (warmup_epochs * epoch_size)
    else:
        lr = initial_lr * (gamma ** (step_index / 2.0))

    if epoch % boost_interval == 0 and epoch > 0:
        lr *= boost_factor

    lr = max(min_lr, min(lr, max_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():
    global optimizer
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    full_dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = data.random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} training, {val_size} validation samples")

    train_loader = data.DataLoader(
        train_dataset, batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=detection_collate
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=detection_collate
    )

    epoch_size = math.ceil(len(train_dataset) / batch_size)
    max_iter = max_epoch * epoch_size
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    start_iter = args.resume_epoch * epoch_size if args.resume_epoch > 0 else 0

    # Early stopping
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(train_loader)
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), os.path.join(save_folder, f"{cfg['name']}_epoch_{epoch}.pth"))
            epoch += 1

        # Unfreeze stage3 + detection heads once the backbone is warmed up.
        if epoch == 2:
            print("Unfreezing stage3 and detection heads (Class/Bbox/Landmark)...")
            body = net.module.body if isinstance(net, torch.nn.DataParallel) else net.body
            for name, param in body.named_parameters():
                param.requires_grad = 'stage3' in name
            for head_name in ['ClassHead', 'BboxHead', 'LandmarkHead']:
                for param in getattr(net, head_name).parameters():
                    param.requires_grad = True
            for module_name in ['fpn', 'ssh1', 'ssh2', 'ssh3']:
                for param in getattr(net, module_name).parameters():
                    param.requires_grad = True

            params = []
            for name, param in net.named_parameters():
                if param.requires_grad:
                    lr = initial_lr * 2.0 if 'ClassHead' in name else initial_lr
                    params.append({'params': [param], 'lr': lr})
            optimizer = optim.SGD(params, momentum=momentum, weight_decay=weight_decay)
            print("Optimizer rebuilt for fine-tuning; classification head gets a higher LR.")

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]

        out = net(images)

        loc_weight = cfg.get('loc_weight', 2.0)
        cls_weight = cfg.get('cls_weight', 2.0)
        landm_weight = cfg.get('landm_weight', 0.5)

        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = loc_weight * loss_l + cls_weight * loss_c + landm_weight * loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        print(f"Epoch:{epoch}/{max_epoch} || Iter:{iteration % epoch_size}/{epoch_size} || "
              f"Loc:{loss_l.item():.4f} Cla:{loss_c.item():.4f} Landm:{loss_landm.item():.4f} "
              f"|| LR:{lr:.6f} || Time:{(load_t1 - load_t0):.2f}s", flush=True)

        # Validation + early stopping at the end of each epoch.
        if (iteration + 1) % epoch_size == 0:
            net.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for val_images, val_targets in val_loader:
                    val_images = val_images.to(device)
                    val_targets = [anno.to(device) for anno in val_targets]
                    val_out = net(val_images)
                    val_loss_l, val_loss_c, val_loss_landm = criterion(val_out, priors, val_targets)
                    val_loss = cfg['loc_weight'] * val_loss_l + val_loss_c + val_loss_landm
                    val_loss_total += val_loss.item()
            val_loss_avg = val_loss_total / len(val_loader)
            print(f"Validation Loss (Epoch {epoch}): {val_loss_avg:.4f}")

            if epoch % 5 == 0:
                try:
                    n_saved = visualize_validation_predictions(
                        net=net, cfg=cfg, val_loader=val_loader, device=device,
                        epoch=epoch, save_dir="./val_visuals", vis_thres=0.3, max_images=10,
                    )
                    print(f"Validation visuals saved: {n_saved}")
                except Exception as e:
                    print("Visualization skipped due to error:", e)

            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                patience_counter = 0
                torch.save(net.state_dict(), os.path.join(save_folder, 'Best_RetinaFace.pth'))
                print(f"New best model saved (val_loss={best_loss:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs.")
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
            net.train()

    torch.save(net.state_dict(), os.path.join(save_folder, 'Final_RetinaFace.pth'))
    print(f"Training complete! Final model saved at: {os.path.join(save_folder, 'Final_RetinaFace.pth')}")


if __name__ == '__main__':
    train()
