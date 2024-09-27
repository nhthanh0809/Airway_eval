import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.ElementTree as ET

from utils.data_utils import *


class AirwayEvaluationDataset(Dataset):
    def __init__(self, img_dir, gt_dir, mask_dir, resize_height, resize_width, point_list, sigma, transform=False, visualization=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.mask_dir = mask_dir
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.img_names = os.listdir(img_dir)
        self.img_nums = len(self.img_names)
        self.point_list = point_list
        self.points_num = len(self.point_list)

        self.sigma = sigma
        self.heatmap_height = int(self.resize_height)
        self.heatmap_width = int(self.resize_width)
        self.visualization = visualization
        self.transform = transform


    def __getitem__(self, i):
        index = i % self.img_nums
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img, scal_ratio_w, scal_ratio_h = self.img_preproccess(img_path, mask_image=False)
        mask, scal_ratio_w, scal_ratio_h = self.img_preproccess(mask_path, mask_image=True)

        ########## Working with xml files ##########
        gt_path = self.gt_dir + 'annotations.xml'

        doc = ET.parse(gt_path)
        root = doc.getroot()
        xml_imageEles = root.findall('image')

        points_name, gt_x, gt_y = self.get_points_from_CVAT_xml(img_name, self.point_list, xml_imageEles)

        # print(scal_ratio_w)
        # print(scal_ratio_h)

        x_all = gt_x / scal_ratio_w
        y_all = gt_y / scal_ratio_h


        # if self.visualization:
        #     visualization(img_path, gt_x, gt_y, points_num=self.points_num, points_name=points_name)
        # if len(x_all) != self.points_num or len(x_all) != self.points_num:
        #     print(img_name)
        heatmaps = self.get_heatmaps(x_all, y_all, self.sigma)
        heatmaps_refine = self.get_refine_heatmaps(x_all / 2, y_all / 2, self.sigma)
        # img = self.data_preproccess(img)
        heatmaps = self.data_preproccess(heatmaps)
        heatmaps_refine = self.data_preproccess(heatmaps_refine)


        return img, mask, heatmaps, heatmaps_refine, img_name, points_name, gt_x, gt_y, scal_ratio_w, scal_ratio_h

    def __len__(self):
        return self.img_nums

    def get_heatmaps(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.points_num, self.heatmap_height, self.heatmap_width))


        for i in range(self.points_num):
            heatmaps[i] = CenterLabelHeatMap(self.heatmap_width, self.heatmap_height, x_all[i], y_all[i], sigma)

            if self.visualization:
                cv2.imshow("Heatmap", heatmaps[i])
                cv2.waitKey(0)
        heatmaps = np.asarray(heatmaps, dtype="float32")

        return heatmaps

    def get_refine_heatmaps(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.points_num, int(self.heatmap_height / 2), int(self.heatmap_width / 2)))
        for i in range(self.points_num):
            heatmaps[i] = CenterLabelHeatMap(int(self.heatmap_width / 2), int(self.heatmap_height / 2), x_all[i],
                                             y_all[i], sigma)
            if self.visualization:
                cv2.imshow("Refine Heatmap", heatmaps[i])
                cv2.waitKey(0)
        heatmaps = np.asarray(heatmaps, dtype="float32")
        return heatmaps

    def img_preproccess(self, img_path, mask_image=False):

        if mask_image:
            img = cv2.imread(img_path, 0)
            img_h, img_w = img.shape
            scal_ratio_w = img_w / self.resize_width
            scal_ratio_h = img_h / self.resize_height
            img = cv2.resize(img, (self.resize_width, self.resize_height))
            img = np.reshape(img, (self.resize_height, self.resize_width, 1)).astype(np.float32)
            # print("mask: ", img.shape)

            img = np.transpose(img, (2, 0, 1))

            img = img.clip(max=1)
            # print(np.amax(img))
            img = torch.from_numpy(img).float()
            # img = img / 255

        else:
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            scal_ratio_w = img_w / self.resize_width
            scal_ratio_h = img_h / self.resize_height
            img = cv2.resize(img, (self.resize_width, self.resize_height))
            # print("img: ", img.shape)
            img = np.transpose(img, (2, 0, 1))


            # print(np.amax(img))

            img = torch.from_numpy(img).float()
            # img = img / 255

            if self.transform:
                # img transform
                transform = transforms.Compose([
                    # transforms.Normalize([121.78, 121.78, 121.78], [74.36, 74.36, 74.36])
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
                )
                img = transform(img)

        return img, scal_ratio_w, scal_ratio_h

    def data_preproccess(self, data):
        data = torch.from_numpy(data).float()
        return data

    def get_points_from_CVAT_xml(self, imageName, points_list, xml_imageEles):
        labels = []
        x_pos = []
        y_pos = []

        for imageEle in xml_imageEles:
            if imageName == imageEle.attrib['name']:
                for i, pointName in enumerate(points_list):
                    for pointEle in imageEle.iter('points'):
                        if pointName == pointEle.attrib['label']:
                            labels.append(pointEle.attrib['label'])
                            x_pos.append(float(pointEle.attrib['points'].split(',')[0]))
                            y_pos.append(float(pointEle.attrib['points'].split(',')[1]))

        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)

        # print(imageName, labels, x_pos, y_pos)

        return labels, x_pos, y_pos


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    # heatmap[int(c_y)][int(c_x)] = 2
    return heatmap

#
#
# if __name__ == '__main__':
#     from configs.configuration import Config
#     from data.points_labels_v2 import *
#     from torch.utils.data import DataLoader
#
#     train_img_dir = 'D:/Self-projects/2D_Cephalometry/src/Ceph-pytorch1.11/output/Total_cuves_20221009_Total_curves/inference_results/images/'
#     gt_train_dir = 'D:/Self-projects/2D_Cephalometry/src/Ceph-pytorch1.11/output/Total_cuves_20221009_Total_curves/inference_results/points/'
#
#
#     point_list = Total_curves
#     train_set = CephaDataset(train_img_dir,
#                              gt_train_dir,
#                              Config.label_type,
#                              Config.resize_h,
#                              Config.resize_w,
#                              point_list,
#                              Config.sigma,
#                              transform=False,
#                              visualization=True)
#
#     train_loader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=True, num_workers=1)
#
#     for i, (img, heatmaps, heatmaps_refine, img_name, x_all, y_all, _, _) in enumerate(train_loader):
#         print(i)

#
#     train_set = CephaDataset(Config.train_img_dir,
#                              Config.gt_train_dir,
#                              Config.label_type,
#                              Config.resize_h,
#                              Config.resize_w,
#                              Config.points_num,
#                              Config.sigma,
#                              Config.transform,
#                              visualization=False)
#
#     valid_set = CephaDataset(Config.train_img_dir,
#                              Config.gt_valid_dir,
#                              Config.label_type,
#                              Config.resize_h,
#                              Config.resize_w,
#                              Config.points_num,
#                              Config.sigma,
#                              visualization=True)
#
#     for data in train_set:
#         print(data)

