from torch.utils.data import *
from imutils import paths
import cv2
import numpy as np
import os
import torch
from datasets.augmentation import ImgAugment_Albumentations
from config import cfg_plate


class labelFpsDataLoader(Dataset):
    def __init__(self, img_dir, preproc):
        self.img_dir = img_dir
        self.img_paths = []
        # for i in range(len(img_dir)):
        #     self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.img_paths = os.listdir(img_dir)
        self.img_paths = [os.path.join(img_dir, x) for x in self.img_paths]
        self.imgaug = ImgAugment_Albumentations(width=cfg_plate['image_size'], height=cfg_plate['image_size'])
        self.preproc = preproc

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)

        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        fps = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        annotations = np.zeros((0, 13))
        bbs = []
        kps = []
        label_box = []
        bbs.append([leftUp[0], leftUp[1], rightDown[0], rightDown[1]])

        # landmarks
        for x, y in fps:
            kps.append((x, y))
        class_labels = [
            'top_left',
            'top_right',
            'bottom_left',
            'bottom_right',
        ]
        label_box.append('plate')

        image_aug, bbs_aug, kps_aug = self.imgaug(img, bbs, kps, label_box, class_labels)
        # print(image_aug.shape)
        debug = False
        if debug:
            img_debug = image_aug.copy()
            img_debug = cv2.cvtColor(img_debug, cv2.COLOR_RGB2BGR)
            for box in bbs_aug:
                b = [int(x) for x in box]
                print("Box: ", b)
                # landm = [int(x) for x in kps_aug.tolist()]
                cv2.rectangle(img_debug, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            for x, y in kps_aug:
                # landms
                print("Landmark: %d %d " % (x, y))
                cv2.circle(img_debug, (int(x), int(y)), 1, (0, 0, 255), 4)
                # cv2.circle(img_debug, (landm[2], landm[3]), 1, (0, 255, 255), 4)
                # cv2.circle(img_debug, (landm[4], landm[5]), 1, (255, 0, 255), 4)
                # cv2.circle(img_debug, (landm[6], landm[7]), 1, (0, 255, 0), 4)

            name = "test_data.jpg"
            cv2.imwrite(name, img_debug)
            img_debug = cv2.imread(name)
            cv2.imshow("test", img_debug)
            cv2.waitKey()
        if len(bbs_aug) == 0 or len(kps_aug) == 0:
            return annotations
        image_aug = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
        annotation = np.zeros((1, 13))
        # bbox
        annotation[0, 0] = round(bbs_aug[0][0], 3)  # x1
        annotation[0, 1] = round(bbs_aug[0][1], 3)  # y1
        annotation[0, 2] = round(bbs_aug[0][2], 3)  # x2
        annotation[0, 3] = round(bbs_aug[0][3], 3)  # y2

        # landmarks
        index = 4
        for x, y in kps_aug:
            annotation[0, index] = round(x, 3)  # x
            index += 1
            annotation[0, index] = round(y, 3)  # y
            index += 1
        if annotation[0, 4] < 0:
            annotation[0, 12] = -1
        else:
            annotation[0, 12] = 1
        annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            image_aug, target = self.preproc(image_aug, target)
        return torch.from_numpy(image_aug), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
    return (torch.stack(imgs, 0), targets)


class labelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        # for i in range(len(img_dir)):
        #     self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.img_paths = os.listdir(img_dir)
        self.img_paths = [os.path.join(img_dir, x) for x in self.img_paths]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].split('.')[0].split('-')[-3]
        return torch.from_numpy(img)
        # return resizedImage, lbl, img_name


class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.long_side = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_raw = cv2.imread(img_name)
        target_size = self.long_side
        im_shape = img_raw.shape
        im_size_min = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        img = cv2.resize(img_raw, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(int(eel) * resize) for eel in el.split('&')] for el in iname[2].split('_')]
        # assert img.shape[0] == 1160

        annotations = np.zeros((0, 4))
        annotation = np.zeros((1, 4))
        # bbox
        annotation[0, 0] = leftUp[0]  # x1
        annotation[0, 1] = leftUp[1]  # y1
        annotation[0, 2] = rightDown[0]  # x2
        annotation[0, 3] = rightDown[1]  # y2
        debug = False
        if debug:
            cv2.rectangle(img, (leftUp[0], leftUp[1]), (rightDown[0], rightDown[1]), (0, 0, 255), 2)

            cv2.imshow("test", img)
            cv2.waitKey()
        annotations = np.append(annotations, annotation, axis=0)
        return torch.from_numpy(img), annotations


if __name__ == "__main__":
    from datasets.data_augment import preproc

    # dst = labelFpsDataLoader("/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_base",
    #                          preproc(512, (104, 117, 123)))
    dst = ChaLocDataLoader(["/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_weather"], imgSize=320)
    # print(len(dst))
    for index in range(0, len(dst)):
        dst.__getitem__(index)
