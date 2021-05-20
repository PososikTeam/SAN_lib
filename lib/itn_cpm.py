import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os



from .model_utils import find_tensor_peak_batch
from .generator_model import define_G

class ITN_CPM(nn.Module):
    def __init__(self, pts_num, stage = 3, argmax = 3):
        super(ITN_CPM, self).__init__()

        self.downsample = 1

        self.netG_A = define_G()
        self.netG_B = define_G()

        self.features = nn.Sequential(
              nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

        self.downsample = 8
        self.pts_num = pts_num
        self.stage = stage
        self.argmax = argmax

        self.CPM_feature = nn.Sequential(
              nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1
              nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2

        self.stage1 = nn.Sequential(
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
              nn.Conv2d(512, pts_num, kernel_size=1, padding=0))

        self.stage2 = nn.Sequential(
              nn.Conv2d(128*2+pts_num*2, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
              nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))

        self.stage3 = nn.Sequential(
              nn.Conv2d(128*2+pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
              nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))


    # return : cpm-stages, locations
    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size = inputs.size(0)

        batch_locs, batch_scos = [], []    # [Squence, Points]

        features, stage1s = [], []
        inputs = [inputs, (self.netG_A(inputs)+self.netG_B(inputs))/2]
        for input in inputs:
            feature  = self.features(input)
            feature = self.CPM_feature(feature)
            features.append(feature)
            stage1s.append( self.stage1(feature) )

        xfeature = torch.cat(features, 1)
        cpm_stage2 = self.stage2(torch.cat([xfeature, stage1s[0], stage1s[1]], 1))
        cpm_stage3 = self.stage3(torch.cat([xfeature, cpm_stage2], 1))


        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(cpm_stage3[ibatch], self.argmax, self.downsample)
            batch_locs.append( batch_location )
            batch_scos.append( batch_score )
            
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

        return batch_locs, batch_scos



def create_san(num_points, path_saved_model = 'san.pth'):
    if os.path.isfile(path_saved_model):
        try:
            loaded_dict = torch.load(path_saved_model)

        except Exception as err:
            return None, None, 'Cant read path \n' + str(err)
        
        model_param = {'stage' : 3, 'argmax' : 3}
        image_convert_param = {'pre_crop_expand' : 0.2, 'crop_width' : 180, 'crop_height' : 180}


        san_model = ITN_CPM(num_points+1, model_param['stage'], model_param['argmax'])

        try:
            san_model.load_state_dict(loaded_dict)
        except Exception as err:
            return None, None, 'weights not match with model \n' + str(err)
       
     

        return san_model, image_convert_param, 'Ok'
    else:
        return None, None, 'Cant finde weights file'
            
            