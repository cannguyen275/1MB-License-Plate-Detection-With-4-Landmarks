import logging
import os
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import random
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


def decode_output(image, detection_boxes, detection_scores, detection_landmark, cfg_plate):
    # print(image.shape[2:])
    image_h, image_w = image.shape[2:]
    # image_h, image_w, _ = image.shape
    # cfg_plate['image_size'] = (480, 640)
    detection_scores = F.softmax(detection_scores, dim=-1)
    # detection_scores = detection_scores.cpu().detach().numpy()
    # priorbox = PriorBox(cfg_plate,
    #                     image_size=(cfg_plate['image_size'], cfg_plate['image_size']), phase='test')  # height, width
    priorbox = PriorBox(cfg_plate,
                        image_size=(image_h, image_w), phase='test')  # height, width
    priors = priorbox.forward()
    priors = priors.to(torch.device('cuda'))
    prior_data = priors.data
    boxes = decode(detection_boxes.data.squeeze(0), prior_data, cfg_plate['variance'])
    # boxes[:, 0::2] = boxes[:, 0::2] * cfg_plate['image_size']  # width
    # boxes[:, 1::2] = boxes[:, 1::2] * cfg_plate['image_size']  # height
    boxes[:, 0::2] = boxes[:, 0::2] * image_w  # width
    boxes[:, 1::2] = boxes[:, 1::2] * image_h  # height
    boxes = boxes.cpu().numpy()
    scores = scores = detection_scores.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(detection_landmark.data.squeeze(0), prior_data, cfg_plate['variance'])
    # landms[:, 0::2] = landms[:, 0::2] * cfg_plate['image_size']
    # landms[:, 1::2] = landms[:, 1::2] * cfg_plate['image_size']
    landms[:, 0::2] = landms[:, 0::2] * image_w
    landms[:, 1::2] = landms[:, 1::2] * image_h
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > cfg_plate['confidence_threshold'])[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:cfg_plate['top_k']]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, cfg_plate['nms_threshold'])
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:cfg_plate['keep_top_k'], :]
    landms = landms[:cfg_plate['keep_top_k'], :]
    dets = np.concatenate((dets, landms), axis=1)
    # draw_ouput2(image, dets)
    return dets


def draw_output(output, image, cfg_efficient, target):
    device = torch.device('cuda')
    image = image.cpu().numpy().transpose(1, 2, 0)
    image += (104, 117, 123)
    image = np.uint8(image)
    cv2.imwrite("test_temp.jpg", image)
    image = cv2.imread('test_temp.jpg')
    detection_boxes, detection_scores, detection_landmark = output
    detection_boxes = detection_boxes.to(device)
    detection_scores = detection_scores.to(device)
    detection_landmark = detection_landmark.to(device)
    dets = decode_output(image, detection_boxes, detection_scores, detection_landmark, cfg_efficient)
    count = 0
    # Draw prediction
    for b in dets:
        color = np.random.randint(255, size=3)
        if b[4] < cfg_efficient['vis_thres']:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (int(color[0]), int(color[1]), int(color[2])), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        # cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
        count += 1
        if count > 10:
            break
    # Draw ground-truth
    boxes = target[:, :4]
    landm = target[:, 4:-1]
    height, width, _ = image.shape
    boxes[:, 0::2] *= width
    boxes[:, 1::2] *= height

    landm[:, 0::2] *= width
    landm[:, 1::2] *= height

    for i, b in enumerate(boxes):
        b = [int(x) for x in b.tolist()]
        b += [1]
        b += [int(x) for x in landm[i].tolist()]
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cx = b[0]
        cy = b[1] + 12
        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[9], b[10]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        # cv2.circle(image, (b[13], b[14]), 1, (0, 255, 0), 4)
    # save image

    name = "test.jpg"
    cv2.imwrite(name, image)


def draw_ouput2(image, dets, save_image=True):
    device = torch.device('cpu')
    image = torch.squeeze(image)
    image = image.cpu().numpy().transpose(1, 2, 0)
    image += (104, 117, 123)
    image = np.uint8(image)
    cv2.imwrite("test_temp.jpg", image)
    image = cv2.imread('test_temp.jpg')
    print(dets)
    # Draw prediction
    for b in dets:
        color = np.random.randint(255, size=3)
        if b[4] < 0.5:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (int(color[0]), int(color[1]), int(color[2])), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        # cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
    if save_image:
        # save image
        temp = random.randint(0, 100000)
        name = os.path.join('results', str(temp) + ".jpg")
        cv2.imwrite(name, image)
        # cv2.imshow(str(temp), image)
        # cv2.waitKey()


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict
