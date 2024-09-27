
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


train_iterations = 0
valid_iterations = 0

def export_training_config(txt_export_path):
    with open(txt_export_path, 'w') as f:
        for item in vars(Config).items():
            line = str(item[0]) + ': ' + str(item[1]) + '\n'
            f.write(line)

def train_model_single_epoch(model, criterion, optimizer, lr_scheduler, train_loader, epoch_num, checkpoint_folder, checkpoint_name, writer, debug_steps=10, device='cpu'):

    model.train()
    training_total_loss = 0.0
    training_segmentation_loss = 0.0
    training_landmark_loss = 0.0
    training_landmark_loss_refine = 0.0

    steps = 0
    for i, (img, mask, gt_heatmap, gt_heatmap_refine, img_name, points_name, gt_x, gt_y, scal_ratio_w, scal_ratio_h) in enumerate(train_loader):

        img = img.to(device)
        mask = mask.to(device)
        gt_heatmaps = gt_heatmap.to(device)
        gt_heatmaps_refine = gt_heatmap_refine.to(device)
        pred_heatmap, pred_heatmap_refine, pred_segment = model(img)


        ### Visualize heatmap during training ###

        if Config.visualize_output_on_training:
            temp_outputs = pred_heatmap[0].cpu().detach().numpy()
            pred = get_predict_point_from_heatmap(temp_outputs, scal_ratio_w, scal_ratio_h, len(temp_outputs), visualization=True)

        landmark_loss = calculate_landmark_loss(criterion, pred_heatmap, gt_heatmaps, Config.base_number)
        landmark_loss_refine = calculate_landmark_loss(criterion, pred_heatmap_refine, gt_heatmaps_refine, Config.base_number)
        segment_loss = calculate_segmentation_loss(pred_segment, mask)

        # print(segment_loss)

        total_loss = Config.loss_weight[0]*landmark_loss + Config.loss_weight[1]*landmark_loss_refine + Config.loss_weight[2]*segment_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        training_total_loss += total_loss.item()
        training_landmark_loss += landmark_loss.item()
        training_landmark_loss_refine += landmark_loss_refine.item()
        training_segmentation_loss += segment_loss.item()

        global train_iterations
        train_iterations +=1

        writer.add_scalar('models/trạin_total_loss', total_loss.cpu().detach().numpy(), train_iterations)
        writer.add_scalar('models/trạin_landmark_loss', landmark_loss.cpu().detach().numpy(), train_iterations)
        writer.add_scalar('models/trạin_landmark_loss_refine', landmark_loss_refine.cpu().detach().numpy(), train_iterations)
        writer.add_scalar('models/trạin_segment_loss', segment_loss.cpu().detach().numpy(), train_iterations)
        writer.add_scalar('models/learning rate', lr_scheduler.get_last_lr()[0], train_iterations)

        steps += 1
        if i and i % debug_steps == 0:

            avg_total_loss = training_total_loss / steps
            avg_landmark_loss = training_landmark_loss / steps
            avg_landmark_loss_refine = training_landmark_loss_refine / steps
            avg_segment_loss = training_segmentation_loss / steps
            print(
                f"Epoch: {epoch_num}, Step: {train_iterations - 1}, " +
                f"Average trạin Total Loss: {avg_total_loss:.4f}, " +
                f"Average trạin Landmark Loss {avg_landmark_loss:.4f}, " +
                f"Average trạin Landmark Loss refine {avg_landmark_loss_refine:.4f}, " +
                f"Average trạin Segment Loss: {avg_segment_loss:.4f}, " +
                f"Learning rate: {lr_scheduler.get_last_lr()[0]:.8f}"
            )

    lr_scheduler.step()

    if epoch % Config.save_weight_every_epoch == 0 or epoch == Config.num_epochs - 1:
        model_path = os.path.join(checkpoint_folder, checkpoint_name + f"-{epoch}.pt")
        torch.save(model.state_dict(), model_path)


def valid_model_by_epoch(model,criterion, valid_loader, epoch_num, writer, device='cpu'):
    model.eval()
    steps = 0
    validation_total_loss = 0.0
    validation_segmentation_loss = 0.0
    validation_landmark_loss = 0.0
    validation_landmark_loss_refine = 0.0

    with torch.no_grad():
        for i, (img, mask, gt_heatmap, gt_heatmap_refine, img_name, points_name, gt_x, gt_y, _, _) in enumerate(valid_loader):

            img = img.to(device)
            mask = mask.to(device)
            gt_heatmaps = gt_heatmap.to(device)
            gt_heatmaps_refine = gt_heatmap_refine.to(device)
            pred_heatmap, pred_heatmap_refine, pred_segment = model(img)

            landmark_loss = calculate_landmark_loss(criterion, pred_heatmap, gt_heatmaps, Config.base_number)
            landmark_loss_refine = calculate_landmark_loss(criterion, pred_heatmap_refine, gt_heatmaps_refine, Config.base_number)
            segment_loss = calculate_segmentation_loss(pred_segment, mask)

            total_loss = landmark_loss + landmark_loss_refine + segment_loss

            validation_total_loss += total_loss.item()
            validation_landmark_loss += landmark_loss.item()
            validation_landmark_loss_refine += landmark_loss_refine.item()
            validation_segmentation_loss += segment_loss.item()



            global valid_iterations
            valid_iterations += 1

            writer.add_scalar('models/Val_total_loss', total_loss.cpu().detach().numpy(), train_iterations)
            writer.add_scalar('models/val_landmark_loss', landmark_loss.cpu().detach().numpy(), train_iterations)
            writer.add_scalar('models/val_landmark_loss_refine', landmark_loss_refine.cpu().detach().numpy(),
                              train_iterations)
            writer.add_scalar('models/val_segment_loss', segment_loss.cpu().detach().numpy(), train_iterations)
            writer.add_scalar('models/learning rate', lr_scheduler.get_last_lr()[0], train_iterations)
            steps += 1

        avg_total_loss = validation_total_loss / steps
        avg_landmark_loss = validation_landmark_loss / steps
        avg_landmark_loss_refine = validation_landmark_loss_refine / steps
        avg_segment_loss = validation_segmentation_loss / steps
        print(
            f"Epoch: {epoch_num}, Step: {train_iterations - 1}, " +
            f"Average Val Total Loss: {avg_total_loss:.4f}, " +
            f"Average Val Landmark Loss {avg_landmark_loss:.4f}, " +
            f"Average Val Landmark Loss refine {avg_landmark_loss_refine:.4f}, " +
            f"Average Val Segment Loss: {avg_segment_loss:.4f}, " +
            f"Learning rate: {lr_scheduler.get_last_lr()[0]:.8f}"
        )


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



    #### write-out training config ####

    txt_config_path = './output/' + experiment_name + '/training_config.txt'
    export_training_config(txt_config_path)

    writer = SummaryWriter(checkpoint_folder + 'logs/')

    train_set = AirwayEvaluationDataset(train_img_dir,
                             gt_train_dir,
                             mask_train_dir,
                             Config.resize_h,
                             Config.resize_w,
                             point_list,
                             Config.sigma,
                             transform=Config.transform,
                             visualization=Config.visualization)
    valid_set = AirwayEvaluationDataset(valid_img_dir,
                             gt_valid_dir,
                             mask_valid_dir,
                             Config.resize_h,
                             Config.resize_w,
                             point_list,
                             Config.sigma,
                             transform=Config.transform,
                             visualization=Config.visualization)



    train_loader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_set, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    # device = 'cpu'
    # print(device)

    model = model.to(device)

    criterion = nn.MSELoss(reduction='none')
    criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, Config.lr_step_milestones, gamma=0.1, last_epoch=-1)

    for epoch in range(0, Config.num_epochs):

        train_model_single_epoch(model,
                                 criterion= criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 train_loader=train_loader,
                                 epoch_num=epoch,
                                 checkpoint_folder=checkpoint_folder,
                                 checkpoint_name= experiment_name,
                                 writer=writer,
                                 debug_steps=Config.debug_steps,
                                 device=device)

        if epoch % Config.validation_step == 0 or epoch == Config.num_epochs - 1:
            valid_model_by_epoch(model,
                                 criterion=criterion,
                                 valid_loader=valid_loader,
                                 epoch_num=epoch,
                                 writer=writer,
                                 device=device)














