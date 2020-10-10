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
        debug = True
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


class labelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
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
        return resizedImage, lbl, img_name


class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

        # tps = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        # for dot in tps:
        #     cv2.circle(img, (int(dot[0]), int(dot[1])), 2, (0, 0, 255), 2)
        # cv2.imwrite("/home/xubb/1_new.jpg", img)

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                      (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]

        resizedImage = resizedImage.astype('float32')
        # Y = Y.astype('int8')
        resizedImage /= 255.0
        # lbl = img_name.split('.')[0].rsplit('-',1)[-1].split('_')[:-1]
        # lbl = img_name.split('/')[-1].split('.')[0].rsplit('-',1)[-1]
        # lbl = map(int, lbl)
        # lbl2 = [[el] for el in lbl]

        # resizedImage = torch.from_numpy(resizedImage).float()
        return resizedImage, new_labels


class demoTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
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
        return resizedImage, img_name


if __name__ == "__main__":
    from datasets.data_augment import preproc

    dst = labelFpsDataLoader("/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_weather",
                             preproc(512, (104, 117, 123)))
    # print(len(dst))
    for index in range(0, len(dst)):
        dst.__getitem__(index)
