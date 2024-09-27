import cv2
import numpy as np
import torch
import xlwt
import os
import time
import random
import json
import os

from configs.configuration import Config
from utils.data_utils import *
import xml.etree.ElementTree as ET
import torch.nn.functional as F
from utils.metrics import *

def Average(lst):
    return sum(lst) / len(lst)

def calculate_evaluation_metrics(model, test_loader, gt_test_dir, mask_test_dir, save_path, point_list, device='cpu' ):

    points_num = len(point_list)
    loss = np.zeros(points_num)



    segment_metrics = BinaryMetrics()


    num_err_below_20 = np.zeros(points_num)
    num_err_below_25 = np.zeros(points_num)
    num_err_below_30 = np.zeros(points_num)
    num_err_below_40 = np.zeros(points_num)

    accuracy_bellow_20 = []
    accuracy_bellow_25 = []
    accuracy_bellow_30 = []
    accuracy_bellow_40 = []

    average_pixel_acc = []
    average_dice = []
    average_precision = []
    average_specificity = []
    average_recall = []


    img_num = 0
    for img_num, (img, mask, gt_heatmap, _, img_name, points_name, gt_x, gt_y, scal_ratio_w, scal_ratio_h) in enumerate(test_loader):
        # print('image: ', img_name[0])
        img = img.to(device)
        mask = mask.to(device)
        gt_heatmaps = gt_heatmap.to(device)

        pred_heatmap, _, pred_segment = model(img)

        pred_heatmap = pred_heatmap[0].cpu().detach().numpy()


        pixel_acc, dice, precision, specificity, recall = segment_metrics(mask, pred_segment)

        average_pixel_acc.append(pixel_acc.cpu().detach().numpy())
        average_dice.append(dice.cpu().detach().numpy())
        average_precision.append(precision.cpu().detach().numpy())
        average_specificity.append(specificity.cpu().detach().numpy())
        average_recall.append(recall.cpu().detach().numpy())

        pred_landmark = get_predict_point_from_heatmap(pred_heatmap, scal_ratio_w, scal_ratio_h, points_num)

        # print(points_name, gt_x, gt_y)


        gt_x = np.trunc(np.reshape(gt_x, (points_num, 1)))
        gt_y = np.trunc(np.reshape(gt_y, (points_num, 1)))
        gt = np.concatenate((gt_x, gt_y), 1)

        for j in range(points_num):
            error = np.sqrt((gt[j][0] - pred_landmark[j][0]) ** 2 + (gt[j][1] - pred_landmark[j][1]) ** 2)
            loss[j] += error
            if error <= 20:
                num_err_below_20[j] += 1
            elif error <= 25:
                num_err_below_25[j] += 1
            elif error <= 30:
                num_err_below_30[j] += 1
            elif error <= 40:
                num_err_below_40[j] += 1

    average_pixel_acc = Average(average_pixel_acc)
    average_dice = Average(average_dice)
    average_precision = Average(average_precision)
    average_specificity = Average(average_specificity)
    average_recall = Average(average_recall)


    loss = loss / (img_num + 1)
    num_err_below_25 = num_err_below_25 + num_err_below_20
    num_err_below_30 = num_err_below_30 + num_err_below_25
    num_err_below_40 = num_err_below_40 + num_err_below_30

    row0 = ['Point names', '<=20', '<=25', '<=30', '<=40', 'mean_err']
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])
    for i in range(0, points_num):
        # point_name = [k for k,v in point_list.items() if v == i+1]
        sheet1.write(i + 1, 0, point_list[i])
        sheet1.write(i + 1, 1, num_err_below_20[i] / (img_num + 1))
        sheet1.write(i + 1, 2, num_err_below_25[i] / (img_num + 1))
        sheet1.write(i + 1, 3, num_err_below_30[i] / (img_num + 1))
        sheet1.write(i + 1, 4, num_err_below_40[i] / (img_num + 1))
        sheet1.write(i + 1, 5, loss[i])
        accuracy_bellow_20.append(num_err_below_20[i] / (img_num + 1))
        accuracy_bellow_25.append(num_err_below_25[i] / (img_num + 1))
        accuracy_bellow_30.append(num_err_below_30[i] / (img_num + 1))
        accuracy_bellow_40.append(num_err_below_40[i] / (img_num + 1))

    sheet1.write(points_num + 1, 0, 'Average')
    sheet1.write(points_num + 1, 1, Average(accuracy_bellow_20))
    sheet1.write(points_num + 1, 2, Average(accuracy_bellow_25))
    sheet1.write(points_num + 1, 3, Average(accuracy_bellow_30))
    sheet1.write(points_num + 1, 4, Average(accuracy_bellow_40))

    sheet1.write(points_num + 2, 0, 'Mean pixel acc')
    sheet1.write(points_num + 2, 1, average_pixel_acc)
    sheet1.write(points_num + 3, 0, 'Mean Dice')
    sheet1.write(points_num + 3, 1, average_dice)
    sheet1.write(points_num + 4, 0, 'Mean Precision')
    sheet1.write(points_num + 4, 1, average_precision)
    sheet1.write(points_num + 5, 0, 'Mean specificity')
    sheet1.write(points_num + 5, 1, average_specificity)
    sheet1.write(points_num + 6, 0, 'Mean recall')
    sheet1.write(points_num + 6, 1, average_recall)


    f.save(save_path)

    return Average(accuracy_bellow_20), average_pixel_acc, average_dice, average_precision, average_specificity, average_recall


def predict(model, input_dir, annotation_path, output_dir, point_list, device='gpu', visualization=False, draw_output_image=False):

    points_num = len(point_list)
    list_img = os.listdir(input_dir + 'images/')
    count = 0

    model.to(device)
    model.eval()

    start_time = time.time()
    for imgName in list_img:
        # print(imgName)
        fileName = imgName.split('.jpeg')[0]
        img_path = input_dir + 'images/' + imgName

        img = cv2.imread(img_path)

        img_h, img_w, _ = img.shape
        img_resize = cv2.resize(img, (Config.resize_w, Config.resize_h))
        output_image = img_resize.copy()
        img_data = np.transpose(img_resize, (2, 0, 1))
        img_data = np.reshape(img_data, (1, 3, Config.resize_h, Config.resize_w))
        img_data = torch.from_numpy(img_data).float()

        scal_ratio_w = img_w / Config.resize_w
        scal_ratio_h = img_h / Config.resize_h

        img_data = img_data.to(device)

        # outputs, _ = models(img_data)
        landmark_outputs, landmark_outputs_refine, segment_output = model(img_data)

        # print('landmark_outputs: ',landmark_outputs.shape)
        # print('landmark_outputs_refine: ', landmark_outputs_refine.shape)
        # print('segment_output: ', segment_output.shape)

        landmark_outputs = landmark_outputs[0].cpu().detach().numpy()

        segment_output = F.sigmoid(segment_output)
        segment_output = segment_output[0][0].cpu().detach().numpy()

        segment_output = segment_output.astype(np.float64) / np.amax(segment_output)  # normalize the data to 0 - 1
        segment_output = 255 * segment_output  # Now scale by 255
        segment_output = segment_output.astype(np.uint8)

        # cv2.imshow("asd", segment_output)
        # cv2.waitKey(0)

        segment_output_resized = cv2.resize(segment_output, (img_w, img_h))

        kernel = np.ones((15, 15), np.uint8)
        segment_output_resized = cv2.erode(segment_output_resized, kernel)

        res = cv2.findContours(segment_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        res_resized = cv2.findContours(segment_output_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        segment_pred = res[-2][-1]
        segment_pred = np.array(segment_pred).squeeze()

        segment_pred_resize = res_resized[-2][-1]
        segment_pred_resize = np.array(segment_pred_resize).squeeze()


        if draw_output_image:
            if (len(segment_pred) > 50):
                output_image = cv2.polylines(output_image, [segment_pred], True, color=(0, 0, 255))
            else:
                print('Error image: ', imgName)
                print(len(segment_pred))

        landmark_pred = get_predict_point_from_heatmap(landmark_outputs, scal_ratio_w, scal_ratio_h, points_num)

        for j in range(points_num):

            x, y = int(landmark_pred[j][0] / scal_ratio_w), int(landmark_pred[j][1] / scal_ratio_h)
            x_org, y_org = (landmark_pred[j][0]), (landmark_pred[j][1])
            point_name = point_list[j]

            point_dict = {}
            point_dict[point_name] = {}
            point_dict[point_name]["x"] = x_org
            point_dict[point_name]["y"] = y_org


            # rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # output_image = cv2.circle(output_image, (x, y), radius=2, color=(0, 0, 255), thickness=2)

            if draw_output_image:
                ########### Draw GT point and predicted points ############
                # cv2.fillPoly(output_image, pts=[segment_pred], color=(0, 255, 0))
                output_image = cv2.circle(output_image, (x, y), radius=1, color=(0, 0, 255), thickness=1)
                output_image = cv2.putText(output_image, str(point_name), (x - 15, y -15), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                                               (255, 0, 0), 1, cv2.LINE_AA)




        output_mask_dir = output_dir + 'masks/'
        output_image_dir = output_dir + 'images/'

        os.makedirs(output_mask_dir, exist_ok=True)
        os.makedirs(output_image_dir, exist_ok=True)

        output_image = cv2.resize(output_image, (img_w, img_h))


        if draw_output_image:
            ########## draw GT point from annotation file ##########
            if os.path.isfile(annotation_path):
                doc = ET.parse(annotation_path)
                root = doc.getroot()
                xml_imageEles = root.findall('image')
                points_name, x_pos, y_pos = get_points_from_CVAT_xml(imgName, point_list, xml_imageEles)
                for i, pointName in enumerate(points_name):
                    output_image = cv2.circle(output_image, (int(x_pos[i]), int(y_pos[i])), radius=1, color=(0, 255, 0),
                                              thickness=1)
                ########## draw GT mask from annotation file ##########
                maskPoints = get_mask_from_CVAT_xml(imgName, xml_imageEles)
                if len(maskPoints) > 50:
                    output_image = cv2.polylines(output_image, [maskPoints], True, color=(0, 255, 0))



        cv2.imwrite(output_image_dir + fileName + '_pred.jpeg', output_image)
        output_mask = cv2.resize(segment_output, (img_w, img_h))
        cv2.imwrite(output_mask_dir + fileName + '_mask.jpeg', output_mask)


        if visualization:
            cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Resized_Window", 1300, 1900)
            cv2.imshow("Resized_Window", output_image)
            cv2.waitKey(0)

        count += 1

    averageFPS = 1.0 / ((time.time() - start_time) / count)
    print("FPS single images: ", averageFPS)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids



