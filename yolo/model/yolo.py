import math
import copy
import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from .backbone import darknet_pan_backbone

def nms(boxes, scores, threshold):
    return torch.ops.torchvision.nms(boxes, scores, threshold)


def batched_nms(boxes, scores, labels, threshold, max_size): # boxes format: (x1, y1, x2, y2)
    offsets = labels.to(boxes) * max_size
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, threshold)
    return keep


def all_batched_nms(ids, boxes, scores, labels, threshold, max_size): # boxes format: (x1, y1, x2, y2)
    offsets = torch.stack((labels, ids, labels, ids), dim=1) * max_size
    boxes_for_nms = boxes + offsets
    keep = nms(boxes_for_nms, scores, threshold)
    return keep


def cxcywh2xyxy(box): # box format: (cx, cy, w, h)
    cx, cy, w, h = box.T
    ws = w / 2
    hs = h / 2
    new_box = torch.stack((cx - ws, cy - hs, cx + ws, cy + hs), dim=1)
    return new_box


def xyxy2cxcywh(box): # box format: (x1, y1, x2, y2)
    x1, y1, x2, y2 = box.T
    new_box = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)
    return new_box




class Head(nn.Module):
    def __init__(self, predictor, anchors, strides,
                 match_thresh, giou_ratio, loss_weights,
                 score_thresh, nms_thresh, detections):
        super().__init__()
        self.predictor = predictor
        self.register_buffer("anchors", torch.Tensor(anchors))
        self.strides = strides

        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        self.loss_weights = loss_weights

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections = detections

        self.merge = False
        self.eval_with_loss = False
        # self.min_size = 2

    def forward(self, features, targets, image_shapes=None, scale_factors=None, max_size=None):
        preds = self.predictor(features)

        if self.training:
            losses = self.compute_loss(preds, targets)
            return losses
        else:
            losses = {}
            if self.eval_with_loss:
                losses = self.compute_loss(preds, targets)

            results = self.inference(preds, image_shapes, scale_factors, max_size)
            return results, losses



    def inference(self, preds, image_shapes, scale_factors, max_size):
        ids, ps, boxes = [], [], []
        for pred, stride, wh in zip(preds, self.strides, self.anchors):  # 3.54s
            pred = torch.sigmoid(pred)
            n, y, x, a = torch.where(pred[..., 4] > self.score_thresh)
            p = pred[n, y, x, a]

            xy = torch.stack((x, y), dim=1)
            xy = (2 * p[:, :2] - 0.5 + xy) * stride
            wh = 4 * p[:, 2:4] ** 2 * wh[a]
            box = torch.cat((xy, wh), dim=1)

            ids.append(n)
            ps.append(p)
            boxes.append(box)

        ids = torch.cat(ids)
        ps = torch.cat(ps)
        boxes = torch.cat(boxes)

        boxes = cxcywh2xyxy(boxes)
        logits = ps[:, [4]] * ps[:, 5:]
        indices, labels = torch.where(logits > self.score_thresh)  # 4.94s
        ids, boxes, scores = ids[indices], boxes[indices], logits[indices, labels]

        results = []
        for i, im_s in enumerate(image_shapes):  # 20.97s
            keep = torch.where(ids == i)[0]  # 3.11s
            box, label, score = boxes[keep], labels[keep], scores[keep]
            # ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1] # 0.27s
            # keep = torch.where((ws >= self.min_size) & (hs >= self.min_size))[0] # 3.33s
            # boxes, objectness, logits = boxes[keep], objectness[keep], logits[keep] # 0.36s

            if len(box) > 0:
                box[:, 0].clamp_(0, im_s[1])  # 0.39s
                box[:, 1].clamp_(0, im_s[0])  # ~
                box[:, 2].clamp_(0, im_s[1])  # ~
                box[:, 3].clamp_(0, im_s[0])  # ~

                keep = batched_nms(box, score, label, self.nms_thresh, max_size)  # 4.43s
                keep = keep[:self.detections]

                nms_box, nms_label = box[keep], label[keep]

                box, label, score = nms_box / scale_factors[i], nms_label, score[keep]  # 0.30s
            results.append(dict(boxes=box, labels=label, scores=score))  # boxes format: (xmin, ymin, xmax, ymax)

        return results


class Transformer(nn.Module):
    def __init__(self, min_size, max_size, stride=32):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride

        self.flip_prob = 0.5
        self.mosaic = False

    def forward(self, images, targets):
        if targets is None:
            transformed = [self.transforms(img, targets) for img in images]
        else:
            targets = copy.deepcopy(targets)
            transformed = [self.transforms(img, tgt) for img, tgt in zip(images, targets)]

        images, targets, scale_factors = zip(*transformed)

        image_shapes = [img.shape[1:] for img in images]

        images = self.batch_images(images)
        return images, targets, scale_factors, image_shapes

    #def normalize(self, image):
    #    mean = image.new(self.mean)[:, None, None]
    #    std = image.new(self.std)[:, None, None]
    #    return (image - mean) / std

    def transforms(self, image, target):
        image, target, scale_factor = self.resize(image, target)

        if self.training:
            if random.random() < self.flip_prob:
                image, target["boxes"] = self.horizontal_flip(image, target["boxes"])

        return image, target, scale_factor

    def horizontal_flip(self, image, boxes):
        w = image.shape[2]
        image = image.flip(2)

        tmp = boxes[:, 0] + 0
        boxes[:, 0] = w - boxes[:, 2]
        boxes[:, 2] = w - tmp
        return image, boxes

    def resize(self, image, target):
        orig_image_shape = image.shape[1:]
        min_size = min(orig_image_shape)
        max_size = max(orig_image_shape)
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)

        if scale_factor != 1:
            size = [round(s * scale_factor) for s in orig_image_shape]
            image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]

            if target is not None:
                box = target["boxes"]
                box[:, [0, 2]] *= size[1] / orig_image_shape[1]
                box[:, [1, 3]] *= size[0] / orig_image_shape[0]
        return image, target, scale_factor

    def batch_images(self, images):
        max_size = tuple(max(s) for s in zip(*(img.shape[1:] for img in images)))
        batch_size = tuple(math.ceil(m / self.stride) * self.stride for m in max_size)

        batch_shape = (len(images), 3,) + batch_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:, :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs



class YOLOv5(nn.Module):
    def __init__(self, num_classes, model_size=(0.33, 0.5),
                 match_thresh=4, giou_ratio=1, img_sizes=(320, 416),
                 score_thresh=0.1, nms_thresh=0.6, detections=100):
        super().__init__()
        # original
        anchors1 = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
        # [320, 416]
        anchors = [
            [[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]],
            [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]],
            [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]],
        ]
        loss_weights = {"loss_box": 0.05, "loss_obj": 1.0, "loss_cls": 0.5}
        
        self.backbone = darknet_pan_backbone(
            depth_multiple=model_size[0], width_multiple=model_size[1]) # 7.5M parameters
        
        in_channels_list = self.backbone.body.out_channels_list
        strides = (8, 16, 32)
        num_anchors = [len(s) for s in anchors]
        predictor = Predictor(in_channels_list, num_anchors, num_classes, strides)
        
        self.head = Head(
            predictor, anchors, strides, 
            match_thresh, giou_ratio, loss_weights, 
            score_thresh, nms_thresh, detections)
        
        if isinstance(img_sizes, int):
            img_sizes = (img_sizes, img_sizes)
        self.transformer = Transformer(
            min_size=img_sizes[0], max_size=img_sizes[1], stride=max(strides))
    
    def forward(self, images, targets=None):
        images, targets, scale_factors, image_shapes = self.transformer(images, targets)
        features = self.backbone(images)
        
        if self.training:
            losses = self.head(features, targets)
            return losses
        else:
            max_size = max(images.shape[2:])
            results, losses = self.head(features, targets, image_shapes, scale_factors, max_size)
            return results, losses
        
    def fuse(self):
        # fusing conv and bn layers
        for m in self.modules():
            if hasattr(m, "fused"):
                m.fuse()


class Predictor(nn.Module):
    def __init__(self, in_channels_list, num_anchors, num_classes, strides):
        super().__init__()
        self.num_outputs = num_classes + 5
        self.mlp = nn.ModuleList()
        
        for in_channels, n in zip(in_channels_list, num_anchors):
            out_channels = n * self.num_outputs
            self.mlp.append(nn.Conv2d(in_channels, out_channels, 1))
            
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        for m, n, s in zip(self.mlp, num_anchors, strides):
            b = m.bias.detach().view(n, -1)
            b[:, 4] += math.log(8 / (416 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (num_classes - 0.99))
            m.bias = nn.Parameter(b.view(-1))
            
    def forward(self, x):
        N = x[0].shape[0]
        L = self.num_outputs
        preds = []
        for i in range(len(x)):
            h, w = x[i].shape[-2:]
            pred = self.mlp[i](x[i])
            pred = pred.permute(0, 2, 3, 1).reshape(N, h, w, -1, L)
            preds.append(pred)
        return preds
    
    