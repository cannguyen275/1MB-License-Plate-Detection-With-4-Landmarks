import utils
import numpy as np
import torch
import torch.nn as nn
import os
import cv2
from tqdm import tqdm
import torchvision.ops as ops
from utils.utils import decode_output
from datasets.ccpd import ChaLocDataLoader, detection_collate
from datasets.data_augment import preproc
from models.basemodel import BaseModel
from config import cfg_plate
import numpy as np

# global index_toSave
index_toSave = 0


def get_detections(img_batch, model, score_threshold=0.5, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        img_batch = img_batch.cpu().numpy()
        img_batch = np.float32(img_batch)
        img_batch -= (104, 117, 123)
        img_batch = img_batch.transpose(0, 3, 1, 2)
        #
        # img_batch = img_batch.to(torch.device('cpu'))
        # img_batch -= torch.tensor([104, 117, 123])
        # img_batch -= torch.tensor([(104, 117, 123)]).cuda()
        # img_batch -= torch.tensor([(104, 117, 123)])
        # img_batch = img_batch.permute(0, 3, 1, 2)
        img_batch = torch.from_numpy(img_batch)
        img_batch = img_batch.type(torch.cuda.FloatTensor)
        # img_batch = img_batch.type(torch.FloatTensor)
        bboxes, classifications, landmarks = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        picked_scores = []

        for i in range(batch_size):
            # classification = torch.exp(classifications[i, :, :])
            classification = classifications[i, :, :]
            # bbox = bboxes[i, :, :].to(torch.device('cpu'))
            # landmark = landmarks[i, :, :].to(torch.device('cpu'))
            bbox = bboxes[i, :, :]
            landmark = landmarks[i, :, :]
            dets = decode_output(img_batch, bbox, classification, landmark, cfg_plate)
            picked_boxes.append(torch.from_numpy(dets[:, :4]))

        return picked_boxes, picked_landmarks, picked_scores


def compute_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    # (N, K) ndarray of overlap between boxes and query_boxes
    return torch.from_numpy(intersection / ua)


def evaluate(val_data, model, threshold=0.5):
    recall = 0.
    precision = 0.
    # for i, data in tqdm(enumerate(val_data)):
    for idx, (images, targets) in enumerate(val_data):
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        # targets = [anno for anno in targets]

        picked_boxes, _, _ = get_detections(images, model)
        recall_iter = 0.
        precision_iter = 0.

        for j, boxes in enumerate(picked_boxes):
            annot_boxes = targets[j]
            annot_boxes = annot_boxes[annot_boxes[:, 0] != -1]

            if boxes.shape[0] == 0 and annot_boxes.shape[0] == 0:
                continue
            elif boxes.shape[0] == 0 and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes.shape[0] != 0 and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.
                continue

            overlap = ops.boxes.box_iou(annot_boxes, boxes.cuda())
            # overlap = ops.boxes.box_iou(annot_boxes, boxes)
            # compute recall
            max_overlap, _ = torch.max(overlap, dim=1)
            mask = max_overlap > threshold
            detected_num = mask.sum().item()
            recall_iter += detected_num / annot_boxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap, dim=0)
            mask = max_overlap > threshold
            true_positives = mask.sum().item()
            precision_iter += true_positives / boxes.shape[0]

        recall += recall_iter / len(picked_boxes)
        precision += precision_iter / len(picked_boxes)

    return recall / len(val_data), precision / len(val_data)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "weights/CCPD/CCPD_150.pth"
    img_dir = [
        "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_weather",
        "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_blur",
        "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_tilt",
        "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_db",
        "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_fn",
        "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_rotate",
        # "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_np",
        "/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_challenge"
    ]
    print("loading model")
    # Initialize model
    model = BaseModel(cfg=cfg_plate)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model.eval()
    model.to(device)
    for i in np.linspace(0.5, 0.9, 8):
        print("############################")
        print("threshold: " + str(i))
        for index, path in enumerate(img_dir):
            print("**************************")
            print(path)
            val_dataset = ChaLocDataLoader([path], imgSize=320)

            valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False,
                                                       num_workers=6, collate_fn=detection_collate, pin_memory=True)

            recall, precision = evaluate(valid_loader, model, threshold=i)
            print(recall, precision)
