# import cv2
# import numpy as np
# import random
# from utils.box_utils import matrix_iof


# def _crop(image, boxes, labels, landm, img_dim):
#     height, width, _ = image.shape
#     pad_image_flag = True

#     for _ in range(250):
#         """
#         if random.uniform(0, 1) <= 0.2:
#             scale = 1.0
#         else:
#             scale = random.uniform(0.3, 1.0)
#         """
#         PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
#         scale = random.choice(PRE_SCALES)
#         short_side = min(width, height)
#         w = int(scale * short_side)
#         h = w

#         if width == w:
#             l = 0
#         else:
#             l = random.randrange(width - w)
#         if height == h:
#             t = 0
#         else:
#             t = random.randrange(height - h)
#         roi = np.array((l, t, l + w, t + h))

#         value = matrix_iof(boxes, roi[np.newaxis])
#         flag = (value >= 1)
#         if not flag.any():
#             continue

#         centers = (boxes[:, :2] + boxes[:, 2:]) / 2
#         mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
#         boxes_t = boxes[mask_a].copy()
#         labels_t = labels[mask_a].copy()
#         landms_t = landm[mask_a].copy()
#         landms_t = landms_t.reshape([-1, 5, 2])

#         if boxes_t.shape[0] == 0:
#             continue

#         image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

#         boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
#         boxes_t[:, :2] -= roi[:2]
#         boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
#         boxes_t[:, 2:] -= roi[:2]

#         # landm
#         landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
#         landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
#         landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
#         landms_t = landms_t.reshape([-1, 10])


# 	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
#         b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
#         b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
#         mask_b = np.minimum(b_w_t, b_h_t) > 0.0
#         boxes_t = boxes_t[mask_b]
#         labels_t = labels_t[mask_b]
#         landms_t = landms_t[mask_b]

#         if boxes_t.shape[0] == 0:
#             continue

#         pad_image_flag = False

#         return image_t, boxes_t, labels_t, landms_t, pad_image_flag
#     return image, boxes, labels, landm, pad_image_flag


# def _distort(image):

#     def _convert(image, alpha=1, beta=0):
#         tmp = image.astype(float) * alpha + beta
#         tmp[tmp < 0] = 0
#         tmp[tmp > 255] = 255
#         image[:] = tmp

#     image = image.copy()

#     if random.randrange(2):

#         #brightness distortion
#         if random.randrange(2):
#             _convert(image, beta=random.uniform(-32, 32))

#         #contrast distortion
#         if random.randrange(2):
#             _convert(image, alpha=random.uniform(0.5, 1.5))

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#         #saturation distortion
#         if random.randrange(2):
#             _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

#         #hue distortion
#         if random.randrange(2):
#             tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
#             tmp %= 180
#             image[:, :, 0] = tmp

#         image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

#     else:

#         #brightness distortion
#         if random.randrange(2):
#             _convert(image, beta=random.uniform(-32, 32))

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#         #saturation distortion
#         if random.randrange(2):
#             _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

#         #hue distortion
#         if random.randrange(2):
#             tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
#             tmp %= 180
#             image[:, :, 0] = tmp

#         image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

#         #contrast distortion
#         if random.randrange(2):
#             _convert(image, alpha=random.uniform(0.5, 1.5))

#     return image


# def _expand(image, boxes, fill, p):
#     if random.randrange(2):
#         return image, boxes

#     height, width, depth = image.shape

#     scale = random.uniform(1, p)
#     w = int(scale * width)
#     h = int(scale * height)

#     left = random.randint(0, w - width)
#     top = random.randint(0, h - height)

#     boxes_t = boxes.copy()
#     boxes_t[:, :2] += (left, top)
#     boxes_t[:, 2:] += (left, top)
#     expand_image = np.empty(
#         (h, w, depth),
#         dtype=image.dtype)
#     expand_image[:, :] = fill
#     expand_image[top:top + height, left:left + width] = image
#     image = expand_image

#     return image, boxes_t


# def _mirror(image, boxes, landms):
#     _, width, _ = image.shape
#     if random.randrange(2):
#         image = image[:, ::-1]
#         boxes = boxes.copy()
#         boxes[:, 0::2] = width - boxes[:, 2::-2]

#         # landm
#         landms = landms.copy()
#         landms = landms.reshape([-1, 5, 2])
#         landms[:, :, 0] = width - landms[:, :, 0]
#         tmp = landms[:, 1, :].copy()
#         landms[:, 1, :] = landms[:, 0, :]
#         landms[:, 0, :] = tmp
#         tmp1 = landms[:, 4, :].copy()
#         landms[:, 4, :] = landms[:, 3, :]
#         landms[:, 3, :] = tmp1
#         landms = landms.reshape([-1, 10])

#     return image, boxes, landms


# def _pad_to_square(image, rgb_mean, pad_image_flag):
#     if not pad_image_flag:
#         return image
#     height, width, _ = image.shape
#     long_side = max(width, height)
#     image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
#     image_t[:, :] = rgb_mean
#     image_t[0:0 + height, 0:0 + width] = image
#     return image_t


# def _resize_subtract_mean(image, insize, rgb_mean):
#     interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
#     interp_method = interp_methods[random.randrange(5)]
#     image = cv2.resize(image, (insize, insize), interpolation=interp_method)
#     image = image.astype(np.float32)
#     image -= rgb_mean
#     return image.transpose(2, 0, 1)


# class preproc(object):

#     def __init__(self, img_dim, rgb_means):
#         self.img_dim = img_dim
#         self.rgb_means = rgb_means

#     def __call__(self, image, targets):
#         assert targets.shape[0] > 0, "this image does not have gt"

#         boxes = targets[:, :4].copy()
#         labels = targets[:, -1].copy()
#         landm = targets[:, 4:-1].copy()

#         image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
#         image_t = _distort(image_t)
#         image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
#         image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
#         height, width, _ = image_t.shape
#         image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
#         boxes_t[:, 0::2] /= width
#         boxes_t[:, 1::2] /= height

#         landm_t[:, 0::2] /= width
#         landm_t[:, 1::2] /= height

#         labels_t = np.expand_dims(labels_t, 1)
#         targets_t = np.hstack((boxes_t, landm_t, labels_t))

#         return image_t, targets_t



import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof


def _crop(image, boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        PRE_SCALES = [0.3, 0.45, 0.6,0.75, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        l = 0 if width == w else random.randrange(width - w)
        t = 0 if height == h else random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t = landms_t.reshape([-1, 10])

        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False
        return image_t, boxes_t, labels_t, landms_t, pad_image_flag

    return image, boxes, labels, landm, pad_image_flag


def _distort(image):
    """Strong color + contrast + hue augmentation."""
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp = np.clip(tmp, 0, 255)
        image[:] = tmp

    image = image.copy()

    if random.random() < 0.4:
        if random.random() < 0.5:
            _convert(image, beta=random.uniform(-40, 40))
        if random.random() < 0.5:
            _convert(image, alpha=random.uniform(0.5, 1.8))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if random.random() < 0.5:
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.6))
        if random.random() < 0.5:
            tmp = image[:, :, 0].astype(int) + random.randint(-20, 20)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:
        if random.random() < 0.4:
            _convert(image, beta=random.uniform(-40, 40))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if random.random() < 0.4:
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.6))
        if random.random() < 0.4:
            tmp = image[:, :, 0].astype(int) + random.randint(-20, 20)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if random.random() < 0.4:
            _convert(image, alpha=random.uniform(0.5, 1.8))

    return image


def _add_gaussian_noise(image, mean=0, sigma=8):
    """Random Gaussian noise for robustness."""
    if random.random() < 0.3:
        noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    return image


def _rotate(image, boxes, landms, angle_range=(-10, 10)):
    """Random small-angle rotation."""
    if random.random() < 0.3:
        angle = random.uniform(*angle_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(104, 117, 123))

        landms = landms.copy().reshape(-1, 5, 2)
        for i in range(len(landms)):
            pts = np.concatenate([landms[i], np.ones((5, 1))], axis=1)
            pts = np.dot(M, pts.T).T
            landms[i] = pts
        landms = landms.reshape(-1, 10)
    return image, boxes, landms


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.random() < 0.5:
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        landms = landms.copy().reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])
    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:height, 0:width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [
        cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
        cv2.INTER_NEAREST, cv2.INTER_LANCZOS4
    ]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


class preproc(object):
    """Main preprocessing pipeline for RetinaFace."""

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()

        # === Augmentation chain ===
        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
        image_t = _distort(image_t)
        image_t = _add_gaussian_noise(image_t)
        image_t, boxes_t, landm_t = _rotate(image_t, boxes_t, landm_t)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)

        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))
        return image_t, targets_t


# import cv2
# import numpy as np
# import random
# from utils.box_utils import matrix_iof


# # ==========================================================
# # Helper: Random Cutout (occlusion simulation)
# # ==========================================================
# def _cutout(image, num_patches=3):
#     h, w, _ = image.shape
#     for _ in range(num_patches):
#         if random.random() < 0.3:
#             size = random.randint(int(0.1 * min(h, w)), int(0.3 * min(h, w)))
#             x1 = random.randint(0, w - size)
#             y1 = random.randint(0, h - size)
#             color = [random.randint(0, 255) for _ in range(3)]
#             image[y1:y1 + size, x1:x1 + size] = color
#     return image


# # ==========================================================
# # Random Crop (keeps at least one face)
# # ==========================================================
# def _crop(image, boxes, labels, landm, img_dim):
#     height, width, _ = image.shape
#     pad_image_flag = True

#     for _ in range(250):
#         PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
#         scale = random.choice(PRE_SCALES)
#         short_side = min(width, height)
#         w = int(scale * short_side)
#         h = w

#         if width == w:
#             l = 0
#         else:
#             l = random.randrange(width - w)
#         if height == h:
#             t = 0
#         else:
#             t = random.randrange(height - h)
#         roi = np.array((l, t, l + w, t + h))

#         value = matrix_iof(boxes, roi[np.newaxis])
#         flag = (value >= 1)
#         if not flag.any():
#             continue

#         centers = (boxes[:, :2] + boxes[:, 2:]) / 2
#         mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
#         boxes_t = boxes[mask_a].copy()
#         labels_t = labels[mask_a].copy()
#         landms_t = landm[mask_a].copy()
#         landms_t = landms_t.reshape([-1, 5, 2])

#         if boxes_t.shape[0] == 0:
#             continue

#         image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

#         boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
#         boxes_t[:, :2] -= roi[:2]
#         boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
#         boxes_t[:, 2:] -= roi[:2]

#         # landmarks
#         landms_t[:, :, :2] -= roi[:2]
#         landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
#         landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
#         landms_t = landms_t.reshape([-1, 10])

#         b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
#         b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
#         mask_b = np.minimum(b_w_t, b_h_t) > 0.0
#         boxes_t = boxes_t[mask_b]
#         labels_t = labels_t[mask_b]
#         landms_t = landms_t[mask_b]

#         if boxes_t.shape[0] == 0:
#             continue

#         pad_image_flag = False
#         return image_t, boxes_t, labels_t, landms_t, pad_image_flag

#     return image, boxes, labels, landm, pad_image_flag


# # ==========================================================
# # Color, Blur, Noise, Rotation, JPEG Distortion
# # ==========================================================
# def _distort(image):

#     def _convert(image, alpha=1, beta=0):
#         tmp = image.astype(float) * alpha + beta
#         tmp[tmp < 0] = 0
#         tmp[tmp > 255] = 255
#         image[:] = tmp

#     image = image.copy()

#     # Random Brightness
#     if random.random() < 0.3:
#         _convert(image, beta=random.uniform(-48, 48))

#     # Random Contrast
#     if random.random() < 0.35:
#         _convert(image, alpha=random.uniform(0.4, 1.8))

#     # HSV Distortion
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     if random.random() < 0.35:
#         _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.8))
#     if random.random() < 0.35:
#         tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
#         tmp %= 180
#         image[:, :, 0] = tmp
#     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

#     # Random Rotation
#     if random.random() < 0.4:
#         angle = random.uniform(-15, 15)
#         h, w = image.shape[:2]
#         M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
#         image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

#     # Random Blur
#     if random.random() < 0.3:
#         ksize = random.choice([3, 5])
#         image = cv2.GaussianBlur(image, (ksize, ksize), 0)

#     # Random Gaussian Noise
#     if random.random() < 0.3:
#         noise = np.random.normal(0, 10, image.shape).astype(np.float32)
#         image = np.clip(image + noise, 0, 255)

#     # Random JPEG Compression
#     if random.random() < 0.3:
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 90)]
#         _, encimg = cv2.imencode('.jpg', image, encode_param)
#         image = cv2.imdecode(encimg, 1)

#     return image


# # ==========================================================
# # Expand Image with Random Color Background
# # ==========================================================
# def _expand(image, boxes, fill, p):
#     if random.randrange(2):
#         return image, boxes

#     height, width, depth = image.shape
#     scale = random.uniform(1, p)
#     w = int(scale * width)
#     h = int(scale * height)

#     left = random.randint(0, w - width)
#     top = random.randint(0, h - height)

#     boxes_t = boxes.copy()
#     boxes_t[:, :2] += (left, top)
#     boxes_t[:, 2:] += (left, top)
#     expand_image = np.empty((h, w, depth), dtype=image.dtype)
#     fill_color = [random.randint(0, 255) for _ in range(3)]
#     expand_image[:, :] = fill_color
#     expand_image[top:top + height, left:left + width] = image
#     return expand_image, boxes_t


# # ==========================================================
# # Horizontal Flip (mirror)
# # ==========================================================
# def _mirror(image, boxes, landms):
#     _, width, _ = image.shape
#     if random.randrange(2):
#         image = image[:, ::-1]
#         boxes = boxes.copy()
#         boxes[:, 0::2] = width - boxes[:, 2::-2]

#         landms = landms.copy().reshape([-1, 5, 2])
#         landms[:, :, 0] = width - landms[:, :, 0]

#         # swap left-right landmarks
#         tmp = landms[:, 1, :].copy()
#         landms[:, 1, :] = landms[:, 0, :]
#         landms[:, 0, :] = tmp
#         tmp1 = landms[:, 4, :].copy()
#         landms[:, 4, :] = landms[:, 3, :]
#         landms[:, 3, :] = tmp1
#         landms = landms.reshape([-1, 10])

#     return image, boxes, landms


# # ==========================================================
# # Padding, Resize, Normalize
# # ==========================================================
# def _pad_to_square(image, rgb_mean, pad_image_flag):
#     if not pad_image_flag:
#         return image
#     height, width, _ = image.shape
#     long_side = max(width, height)
#     image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
#     image_t[:, :] = rgb_mean
#     image_t[0:height, 0:width] = image
#     return image_t


# def _resize_subtract_mean(image, insize, rgb_mean):
#     interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
#     interp_method = random.choice(interp_methods)
#     image = cv2.resize(image, (insize, insize), interpolation=interp_method)
#     image = image.astype(np.float32)
#     image -= rgb_mean
#     return image.transpose(2, 0, 1)


# # ==========================================================
# # Main Preprocessing Pipeline
# # ==========================================================
# class preproc(object):

#     def __init__(self, img_dim, rgb_means):
#         self.img_dim = img_dim
#         self.rgb_means = rgb_means

#     def __call__(self, image, targets):
#         assert targets.shape[0] > 0, "this image does not have gt"

#         boxes = targets[:, :4].copy()
#         labels = targets[:, -1].copy()
#         landm = targets[:, 4:-1].copy()

#         # 1️⃣ Random Crop
#         image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)

#         # 2️⃣ Color / Blur / Rotation / Noise / Compression
#         image_t = _distort(image_t)

#         # 3️⃣ Random Expand
#         image_t, boxes_t = _expand(image_t, boxes_t, self.rgb_means, 4.0)

#         # 4️⃣ Pad if needed
#         image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)

#         # 5️⃣ Mirror
#         image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)

#         # 6️⃣ Random Cutout
#         image_t = _cutout(image_t)

#         # 7️⃣ Resize + Normalize
#         height, width, _ = image_t.shape
#         image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

#         boxes_t[:, 0::2] /= width
#         boxes_t[:, 1::2] /= height
#         landm_t[:, 0::2] /= width
#         landm_t[:, 1::2] /= height

#         labels_t = np.expand_dims(labels_t, 1)
#         targets_t = np.hstack((boxes_t, landm_t, labels_t))
#         return image_t, targets_t

