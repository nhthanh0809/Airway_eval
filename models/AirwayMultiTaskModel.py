from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummaryX import summary
from torchvision import models, datasets, transforms
from torch.nn import functional as F
from models.LandmarkDetection_head import LandmarkDetectionHead
from models.Segmentation_head import Unet


#########################################################################
# Reference source code:
# https://github.com/mkisantal/backboned-unet/blob/master/backboned_unet/unet.py
# https://github.com/usuyama/pytorch-unet
# https://github.com/mberkay0/pretrained-backbones-unet
########################################################################


class AirwayMultiTaskModel(nn.Module):
    def __init__(self, points_num=17, classes=1):
        super(AirwayMultiTaskModel, self).__init__()

        self.backbone_name = 'densenet121'
        self.features = models.densenet121(pretrained=True).features

        # for param in self.features.parameters():
        #     param.requires_grad = True

        self.feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        self.backbone_output = 'denseblock4'

        self.landmarkDetectionHead = LandmarkDetectionHead(points_num)
        self.segmentationHead = Unet(classes=classes, shortcut_features=self.feature_names, bb_out_name=self.backbone_output)


    def forward_landmarkDetection(self, x):
        w1_f0 = self.landmarkDetectionHead.w1_conv11_0(x)

        x = self.features[0](x)
        w1_f1 = x
        # print("features[0]: ", w1_f1.shape)

        for i in range(1, 5):
            x = self.features[i](x)
        w1_f2 = x
        # print("features[4]: ", w1_f2.shape)

        for i in range(5, 7):
            x = self.features[i](x)
        w1_f3 = x
        # print("features[6]: ", w1_f3.shape)
        for i in range(7, 9):
            x = self.features[i](x)
        w1_f4 = x
        # print("features[8]: ", w1_f4.shape)

        for i in range(9, 12):
            x = self.features[i](x)

        # print("features[11]: ", x.shape)

        x = self.landmarkDetectionHead.w1_conv33_01(x)
        x = self.landmarkDetectionHead.w1_conv11_1(x)
        x = self.landmarkDetectionHead.upsample2(x)
        x = torch.cat((x, w1_f4), 1)
        x = self.landmarkDetectionHead.w1_conv11_2(x)
        x = self.landmarkDetectionHead.upsample2(x)
        x = torch.cat((x, w1_f3), 1)
        x = self.landmarkDetectionHead.w1_conv11_3(x)
        x = self.landmarkDetectionHead.upsample2(x)
        x = torch.cat((x, w1_f2), 1)
        x = self.landmarkDetectionHead.mid_conv11_1(x)
        x = self.landmarkDetectionHead.mid_conv33_01(x)
        x = self.landmarkDetectionHead.mid_conv11_2(x)
        x = self.landmarkDetectionHead.upsample2(x)
        x = torch.cat((x, w1_f1), 1)
        x = self.landmarkDetectionHead.mid_conv11_3(x)

        x = self.landmarkDetectionHead.w2_conv11_5(x)

        refine_hp = self.landmarkDetectionHead.conv_33_refine1(x)
        refine_hp = self.landmarkDetectionHead.conv_11_refine(refine_hp)

        x = self.landmarkDetectionHead.upsample2(x)
        refine1_up = self.landmarkDetectionHead.upsample2(refine_hp)
        x = torch.cat((x, w1_f0, refine1_up), 1)


        # output
        hp = self.landmarkDetectionHead.conv_33_last1(x)
        hp = self.landmarkDetectionHead.conv_11_last(hp)

        return hp, refine_hp

    def forward_segmentation(self, x):

        """ Forward propagation in U-Net. """
        features = {None: None} if None in self.segmentationHead.shortcut_features else dict()
        for name, child in self.features.named_children():
            x = child(x)
            if name in self.segmentationHead.shortcut_features:
                features[name] = x
            if name == self.segmentationHead.bb_out_name:
                break

        for skip_name, upsample_block in zip(self.segmentationHead.shortcut_features[::-1], self.segmentationHead.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.segmentationHead.final_conv(x)

        return x

    def forward(self, x):

        pred_heatmap, pred_heatmap_refine = self.forward_landmarkDetection(x)
        pred_segment = self.forward_segmentation(x)

        return pred_heatmap, pred_heatmap_refine, pred_segment


if __name__ == '__main__':
    # # img = torch.rand([1, 3, 512, 480])
    # # models = CephaLandmark_v2(points_num=17)
    # # outputs, outputs_refine = models(img)
    #
    #
    # features = models.densenet121(pretrained=True).features
    #
    # for param in features.parameters():
    #     param.requires_grad = False
    #
    # # x = feature[0](img)
    #
    # # for i in range (1,5):
    # #     print("feature[" + str(i) + ")]:  " + str(feature[i]))
    #
    # # arch = summary(feature, img)
    # # arch = summary(feature[1:5],x)

    img = torch.rand([1, 3, 512, 480])
    airwayEvaluation_model = AirwayMultiTaskModel(points_num=17, classes=1)

    landmark_hp, landmark_refine_hp, segment_output = airwayEvaluation_model(img)
    # print(landmark_hp.shape)
    # print(landmark_refine_hp.shape)
    # print(segment_output.shape)















