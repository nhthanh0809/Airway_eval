
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

    data_dir = Config.root_data_dir + data_version + '/'

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

    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(evaluation_result_folder, exist_ok=True)

    os.makedirs(best_checkpoint_folder, exist_ok=True)
    os.makedirs(best_checkpoint_evaluation, exist_ok=True)
    os.makedirs(best_checkpoint_inference_folder, exist_ok=True)

    test_set = AirwayEvaluationDataset(test_img_dir,
                                        gt_test_dir,
                                        mask_test_dir,
                                        Config.resize_h,
                                        Config.resize_w,
                                        point_list,
                                        Config.sigma,
                                        transform=Config.transform,
                                        visualization=Config.visualization)


    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=4)



    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    evaluation_results = []
    checkpoint_name = []
    checkpoint_list = os.listdir(checkpoint_folder)
    for checkpoint in checkpoint_list:
        if os.path.isfile(checkpoint_folder + checkpoint):
            fileName = checkpoint.split('.pt')[0]
            output_filePath = evaluation_result_folder + fileName + '.xls'

            # models = torch.load(Config.checkpoint_folder + checkpoint, map_location=device)
            model.load_state_dict(torch.load(checkpoint_folder + checkpoint, map_location=device))
            model.eval()

            print("Evaluating models: " + checkpoint)
            avg_landmark_acc, avg_pixel_acc, avg_dice, avg_precision, avg_specificity, avg_recall = calculate_evaluation_metrics(model, test_loader, gt_test_dir, mask_test_dir, output_filePath, Airway_list, device)
            print("Average landmark accuracy: ", avg_landmark_acc)
            print("Average pixel accuracy: ", avg_pixel_acc)
            print("Average dice score: ", avg_dice)
            print("Average precision: ", avg_precision)
            print("Average specificity: ", avg_specificity)
            print("Average recall: ", avg_recall)
            print("======" * 30)
            evaluation_results.append(avg_landmark_acc)
            checkpoint_name.append(checkpoint)
        else:
            print("There is no checkpoint file here!")

    max_value = max(evaluation_results)
    max_index = evaluation_results.index(max_value)
    print("Best models is: ", checkpoint_name[max_index])

    #### Move best checkpoint to best_checkpoint_folder #####

    import shutil

    src_ckp = checkpoint_folder + checkpoint_name[max_index]
    dst_ckp = best_checkpoint_folder + checkpoint_name[max_index]

    shutil.copy(src_ckp, dst_ckp)

    xls_fileName = checkpoint_name[max_index].split('.pt')[0] + '.xls'
    src_xls = evaluation_result_folder + xls_fileName
    dst_xls = best_checkpoint_evaluation + xls_fileName

    shutil.copy(src_xls, dst_xls)
    #### Delete other checkpoint #####
    for checkpoint in checkpoint_list:
        if os.path.isfile(checkpoint_folder + checkpoint):
            os.remove(checkpoint_folder + checkpoint)

