
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from configs.configuration import Config
from data.dataloader import AirwayEvaluationDataset
from models.losses import *
from data.points_labels import *
from utils.utils import *
from models.AirwayMultiTaskModel import AirwayMultiTaskModel
from utils.data_utils import *
import numpy as np
import torch
import os
import torch.nn as nn
import time


if __name__ == '__main__':

    point_list = Airway_list
    data_version = Config.data_version
    model = AirwayMultiTaskModel(points_num=len(point_list), classes=1)

    data_dir = '/media/prdcv193/Data1/ThanhNH/Ceph/data/Airway/public/v1.0/test/'
    input_dir = '/media/prdcv193/Data1/ThanhNH/Ceph/data/Airway/public/v1.0/test/'
    annotation_path = '/media/prdcv193/Data1/ThanhNH/Ceph/data/Airway/public/v1.0/annotations.xml'

    train_img_dir = data_dir + 'train/images/'
    valid_img_dir = data_dir + 'test/images/'
    test_img_dir = data_dir + 'test/images/'

    mask_train_dir = data_dir + 'train/masks/'
    mask_test_dir = data_dir + 'test/masks/'
    mask_valid_dir = data_dir + 'test/masks/'

    gt_train_dir = data_dir
    gt_valid_dir = data_dir
    gt_test_dir = data_dir

    experiment_name = data_version + Config.root_experiment_name

    checkpoint_folder = './output/' + experiment_name + '/trained_models/'
    evaluation_result_folder = './output/' + experiment_name + '/evaluation_results/'

    best_checkpoint_folder = './output/' + experiment_name + '/best_model/'
    best_checkpoint_evaluation = './output/' + experiment_name + '/best_model/evaluation_results/'
    best_checkpoint_inference_folder = './output/' + experiment_name + '/best_model/inference_results/'

    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    for file in os.listdir(best_checkpoint_folder):
        if file.endswith('.pt'):
            print(file)
            fileName = file.split('.pt')[0]
            # print(fileName)
            output_filePath = evaluation_result_folder + fileName + '.xls'

            # models = torch.load(Config.checkpoint_folder + checkpoint, map_location=device)
            model.load_state_dict(torch.load(best_checkpoint_folder + file, map_location=device))
            model.eval()

            predict(model, input_dir, annotation_path, best_checkpoint_inference_folder, point_list, device, visualization=False,
                    draw_output_image=True)













